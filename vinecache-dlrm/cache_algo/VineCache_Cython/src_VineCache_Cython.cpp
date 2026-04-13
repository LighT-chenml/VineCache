#include "VineCache_Cython.hpp"

using namespace std;

// --- Constants ---
constexpr int N_GROUPS = 26;
constexpr int N_LEVELS = 30;
constexpr int EMBEDDING_SIZE = 36;
constexpr int BYTES_PER_EMBEDDING = 144; // 36 floats * 4 bytes

// --- RL Configuration ---
constexpr int BUCKETS_FUNC = 2;
constexpr int BUCKETS_MMR = 10;
constexpr int BUCKETS_AGING = 10;
constexpr int BUCKETS_SKIP = 10;
constexpr int BUCKETS_LEVEL = 30;
constexpr int STATE_SPACE_SIZE = BUCKETS_FUNC * BUCKETS_MMR * BUCKETS_AGING * BUCKETS_SKIP * BUCKETS_LEVEL;
constexpr int ACTION_SPACE_SIZE = 54;

// --- Cache Configuration ---
int cache_capacity = -1;
double priority_rates[3] = {0.4, 0.8, 0.95};
int hit_thresholds[3];

// --- Cache State ---
int current_size = 0;
int global_clock = 0;
int recent_evictions = 0;
int total_level_sum = 0;
list<Cache_data> level_lists[N_LEVELS];
unordered_map<string, list<Cache_data>::iterator> experience_map;

// --- RL Parameters ---
int exp_func_type = 0;
int age_factor = 100;
double mmr_rate = 0.3;
int aging_threshold;
int random_skip_threshold = 50;

// --- Performance Tracking ---
int total_group_hits = 0;
int last_total_hits = 0;
int last_period_hits = 0;
vector<double> hit_latency_stats[N_GROUPS + 1];
vector<int> ev_file_descriptors(N_GROUPS);
string EV_TABLE_PATH = "/mnt/workssd/xxx/stored_model/criteo_kaggle_all_mmap/epoch-00/ev-table/binary/";

// --- RL State ---
double q_table[STATE_SPACE_SIZE][ACTION_SPACE_SIZE];
int last_state_idx = -1;
int last_action_idx = -1;
int tuning_frequency = 10000;
double learning_rate = 0.1;
double discount_factor = 0.99;
double exploration_rate = 0.1;
double exploration_min = 0.01;
double exploration_decay = 0.001;

// --- Adjustment Steps ---
double mmr_step;
int aging_step;
int skip_step;

/**
 * @brief Initialize the cache with a specific capacity.
 */
void init(int capacity) {
    srand(42); // Standard seed for reproducibility
    
    cache_capacity = capacity;
    aging_threshold = static_cast<int>(capacity * 0.1);
    
    // Clear data structures
    for (int i = 0; i < N_LEVELS; i++) {
        level_lists[i].clear();
    }
    experience_map.clear();
    
    // Initialize hit thresholds
    for (int i = 0; i < 3; i++) {
        hit_thresholds[i] = static_cast<int>(priority_rates[i] * N_GROUPS);
    }
    
    // Initialize learning step sizes
    mmr_step = mmr_rate * 0.05;
    aging_step = aging_threshold * 0.05;
    skip_step = random_skip_threshold * 0.05;
    
    // Reset performance counters
    current_size = 0;
    global_clock = 0;
    recent_evictions = 0;
    total_group_hits = 0;
    last_total_hits = 0;
    last_period_hits = 0;
    total_level_sum = 0;
    
    // Initialize RL table and state
    memset(q_table, 0, sizeof(q_table));
    last_state_idx = -1;
    last_action_idx = -1;
}

/**
 * @brief Load embedding table binary files into file descriptors.
 */
void load_ev_tables() {
    for (int i = 0; i < N_GROUPS; i++) {
        string table_path = EV_TABLE_PATH + "ev-table-" + to_string(i + 1) + ".bin";
        printf("[IO] Loading embedding table: %s\n", table_path.c_str());
        
        int fd = open(table_path.c_str(), O_RDONLY | O_DIRECT);
        if (fd != -1) {
            ev_file_descriptors[i] = fd;
        } else {
            fprintf(stderr, "[Error] Failed to open embedding table: %s\n", table_path.c_str());
        }
    }
}

/**
 * @brief Close embedding table files and report performance metrics.
 */
void close_ev_tables() {
    // Close file descriptors
    for (int i = 0; i < N_GROUPS; i++) {
        if (ev_file_descriptors[i] != -1) {
            close(ev_file_descriptors[i]);
            ev_file_descriptors[i] = -1;
        }
    }
    
    // Performance aggregation
    double total_ms = 0;
    int total_requests = 0;
    
    printf("\n--- Cache Performance Report ---\n");
    for (int i = 0; i <= N_GROUPS; i++) {
        if (!hit_latency_stats[i].empty()) {
            double group_sum_ms = 0;
            for (double lat : hit_latency_stats[i]) {
                group_sum_ms += lat;
            }
            double avg_ms = group_sum_ms / hit_latency_stats[i].size();
            total_requests += hit_latency_stats[i].size();
            total_ms += group_sum_ms;
            
            printf("Hits: %2d | Requests: %6zu | Avg Latency: %.4f ms\n", 
                   i, hit_latency_stats[i].size(), avg_ms);
        }
    }
    
    if (total_requests > 0) {
        printf("--------------------------------\n");
        printf("Overall: %d requests | Average Latency: %.4f ms\n", 
               total_requests, total_ms / total_requests);
        printf("--------------------------------\n\n");
    }
}

/**
 * @brief Read an embedding vector from disk.
 */
vector<float> read_embedding_from_file(int table_id, int row_id) {
    vector<float> embedding(EMBEDDING_SIZE);
    int fd = ev_file_descriptors[table_id - 1];
    
    if (fd == -1) {
        fprintf(stderr, "[Error] Invalid FD for table %d\n", table_id);
        return embedding;
    }
    
    char buffer[BYTES_PER_EMBEDDING];
    off_t offset = static_cast<off_t>(BYTES_PER_EMBEDDING) * row_id;
    
    if (pread(fd, buffer, BYTES_PER_EMBEDDING, offset) != BYTES_PER_EMBEDDING) {
        fprintf(stderr, "[Error] Failed to read embedding from table %d, row %d\n", table_id, row_id);
        return embedding;
    }
    
    // Convert buffer bits to floats
    memcpy(embedding.data(), buffer, BYTES_PER_EMBEDDING);
    return embedding;
}

/**
 * @brief Determine the hierarchy level based on experience value.
 */
int determine_level(double experience) {
    if (experience <= 0) return 0;
    int exp_int = static_cast<int>(experience);
    int bits = 0;
    while (exp_int > 0) {
        exp_int >>= 1;
        bits++;
    }
    return max(0, min(N_LEVELS - 1, bits - 2));
}

/**
 * @brief Calculate base reward/experience delta based on group hit count.
 */
double calculate_reward_base(int hit_count) {
    if (exp_func_type == 0) {
        // Linear piecewise reward
        if (hit_count < hit_thresholds[0]) return 0;
        if (hit_count < hit_thresholds[1]) return hit_count - hit_thresholds[0];
        
        if (hit_count < hit_thresholds[2]) {
            return (hit_thresholds[1] - hit_thresholds[0]) + (hit_count - hit_thresholds[1]) * 2.0;
        }
        
        return (hit_thresholds[1] - hit_thresholds[0]) + 
               (hit_thresholds[2] - hit_thresholds[1]) * 2.0 + 
               (hit_count - hit_thresholds[2]) * 8.0;
    } else {
        // Exponential reward
        return exp((hit_count - 10) / 2.0);
    }
}

/**
 * @brief Standard sigmoid implementation.
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Calculate MMR factor based on level variance.
 */
double calculate_mmr_factor(int variance) {
    double score = 0;
    if (variance > 0) score = log(static_cast<double>(variance));
    else if (variance < 0) score = -log(static_cast<double>(-variance));
    
    return sigmoid(score * mmr_rate);
}

/**
 * @brief Compute the current state index for the RL agent.
 */
int compute_state_index() {
    int f_idx = exp_func_type;
    int m_idx = static_cast<int>(mmr_rate / 0.1);
    int a_idx = static_cast<int>(aging_threshold / (cache_capacity * 0.01));
    int s_idx = static_cast<int>(random_skip_threshold / 10);
    int l_idx = experience_map.empty() ? 0 : static_cast<int>(total_level_sum / experience_map.size());
    
    return f_idx * (BUCKETS_MMR * BUCKETS_AGING * BUCKETS_SKIP * BUCKETS_LEVEL) +
           max(0, min(9, m_idx)) * (BUCKETS_AGING * BUCKETS_SKIP * BUCKETS_LEVEL) +
           max(0, min(9, a_idx)) * (BUCKETS_SKIP * BUCKETS_LEVEL) +
           max(0, min(9, s_idx)) * BUCKETS_LEVEL +
           max(0, min(29, l_idx));
}

/**
 * @brief Find the action with the highest Q-value for a given state.
 */
int find_best_action(int state_idx) {
    int best_idx = 0;
    double max_val = q_table[state_idx][0];
    
    for (int i = 1; i < ACTION_SPACE_SIZE; i++) {
        if (q_table[state_idx][i] > max_val) {
            max_val = q_table[state_idx][i];
            best_idx = i;
        }
    }
    return best_idx;
}

/**
 * @brief Apply the selected action to adjust cache parameters.
 */
void apply_action(int action_idx) {
    int func_act = action_idx / 27;
    int mmr_act = (action_idx / 9) % 3;
    int aging_act = (action_idx / 3) % 3;
    int skip_act = action_idx % 3;
    
    // Toggle experience function
    if (func_act == 1) exp_func_type ^= 1;
    
    // Adjust MMR rate
    if (mmr_act == 1) mmr_rate = max(0.1, mmr_rate - mmr_step);
    else if (mmr_act == 2) mmr_rate = min(1.0, mmr_rate + mmr_step);
    
    // Adjust aging threshold
    if (aging_act == 1) aging_threshold = max(0, aging_threshold - aging_step);
    else if (aging_act == 2) aging_threshold = min(static_cast<int>(cache_capacity * 0.1), aging_threshold + aging_step);
    
    // Adjust random skip threshold
    if (skip_act == 1) random_skip_threshold = max(0, random_skip_threshold - skip_step);
    else if (skip_act == 2) random_skip_threshold = min(100, random_skip_threshold + skip_step);
}

/**
 * @brief Select an action using the ε-greedy strategy.
 */
int select_action(int state_idx) {
    if (static_cast<double>(rand()) / RAND_MAX < exploration_rate) {
        return rand() % ACTION_SPACE_SIZE;
    }
    return find_best_action(state_idx);
}

/**
 * @brief Calculate the reward based on the change in group hit count.
 */
double compute_reward() {
    int current_hits = total_group_hits - last_total_hits;
    int reward = current_hits - last_period_hits;
    
    last_total_hits = total_group_hits;
    last_period_hits = current_hits;
    return static_cast<double>(reward);
}

/**
 * @brief Execute the Q-learning fine-tuning cycle.
 */
void fine_tune_parameters() {
    int new_state_idx = compute_state_index();
    double reward = compute_reward();
    
    // Update Q-table if we have a previous state
    if (last_state_idx != -1) {
        int best_next_idx = find_best_action(new_state_idx);
        double current_q = q_table[last_state_idx][last_action_idx];
        double max_future_q = q_table[new_state_idx][best_next_idx];
        
        q_table[last_state_idx][last_action_idx] += learning_rate * (reward + discount_factor * max_future_q - current_q);
    }
    
    // Select and apply new action
    int next_action_idx = select_action(new_state_idx);
    apply_action(next_action_idx);
    
    last_state_idx = new_state_idx;
    last_action_idx = next_action_idx;
    
    // Decay exploration
    exploration_rate = max(exploration_min, exploration_rate - exploration_decay);
}

/**
 * @brief Evict an entry from the cache using a random walk LRU strategy.
 */
void evict_entry() {
    recent_evictions++;
    
    int target_level = -1;
    int earliest_ts = INT_MAX;
    
    // Find eviction candidate by sampling levels
    for (int i = 0; i < N_LEVELS; i++) {
        if (level_lists[i].empty()) continue;
        
        int ts = level_lists[i].back().last_access_timestamp;
        if (ts < earliest_ts) {
            earliest_ts = ts;
            target_level = i;
        }
        
        // Random skip to explore other levels
        if ((rand() % 1000) < random_skip_threshold) continue;
        break;
    }
    
    if (target_level != -1) {
        string key_to_evict = level_lists[target_level].back().key;
        level_lists[target_level].pop_back();
        experience_map.erase(key_to_evict);
        
        current_size--;
        total_level_sum -= target_level;
    }
}

/**
 * @brief Insert a new embedding into the cache.
 */
void cache_insert(const string& key, vector<float>* val_ptr, double exp) {
    if (current_size >= cache_capacity) {
        evict_entry();
    }
    
    int level = determine_level(exp);
    level_lists[level].push_front(Cache_data(key, val_ptr, exp, global_clock));
    experience_map[key] = level_lists[level].begin();
    
    current_size++;
    total_level_sum += level;
}

/**
 * @brief Handle a cache hit by updating experience and hierarchy level.
 */
void cache_hit(const string& key, double exp_delta) {
    auto it = experience_map.find(key);
    if (it == experience_map.end()) return;
    
    double old_exp = it->second->exp;
    double new_exp = old_exp + exp_delta;
    int old_level = determine_level(old_exp);
    int new_level = determine_level(new_exp);
    
    if (old_level != new_level) {
        // Move to the top of the new level
        vector<float>* val_ptr = it->second->embedding_value;
        level_lists[old_level].erase(it->second);
        level_lists[new_level].push_front(Cache_data(key, val_ptr, new_exp, global_clock));
        experience_map[key] = level_lists[new_level].begin();
        total_level_sum += (new_level - old_level);
    } else {
        // Refresh position in same level
        it->second->exp = new_exp;
        it->second->last_access_timestamp = global_clock;
        level_lists[old_level].splice(level_lists[old_level].begin(), level_lists[old_level], it->second);
    }
}

/**
 * Generate cache key for table and row
 */
string generate_cache_key(int table_id, int row_id) {
    return to_string(table_id) + "-" + to_string(row_id);
}

/**
 * Main request processing function for VineCache cache
 */
/**
 * @brief Process a group request (batch of embedding lookups).
 */
void request_to_vinecache(vector<int>& group_keys, int& group_total_hit, int& group_perfect_hit, 
                       vector<vector<float>>& embeddings, bool is_warmup, bool use_gpu) {
    
    struct timespec start_ts, end_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    
    vector<double> exp_deltas(N_GROUPS);
    vector<int> levels(N_GROUPS);
    vector<bool> lookup_hits(N_GROUPS, false);
    vector<string> cache_keys(N_GROUPS);
    
    global_clock++;
    int total_hits = 0;
    long long l_sum = 0, l_sum2 = 0, l_sum3 = 0;
    
    // Phase 1: Cache lookup and miss handling
    for (int i = 0; i < N_GROUPS; i++) {
        string key = generate_cache_key(i + 1, group_keys[i]);
        cache_keys[i] = key;
        
        auto it = experience_map.find(key);
        if (it != experience_map.end()) {
            // Cache Hit
            embeddings[i] = *(it->second->embedding_value);
            lookup_hits[i] = true;
            total_hits++;
            levels[i] = determine_level(it->second->exp);
            
            // Accumulate statistics for MMR
            l_sum += levels[i];
            l_sum2 += static_cast<long long>(levels[i]) * levels[i];
            l_sum3 += static_cast<long long>(levels[i]) * levels[i] * levels[i];
        } else {
            // Cache Miss
            embeddings[i] = read_embedding_from_file(i + 1, group_keys[i]);
            lookup_hits[i] = false;
            group_perfect_hit = 0;
        }
    }
    
    // Update hit statistics
    if (total_hits == N_GROUPS) {
        total_group_hits++;
    }
    group_hit_total = total_hits;
    
    // Phase 2: Calculate Experience Deltas using MMR
    double reward_base = calculate_reward_base(total_hits) * age_factor / 100.0;
    for (int i = 0; i < N_GROUPS; i++) {
        int l = levels[i];
        long long variance = l_sum3 - 3 * l * l_sum2 + 3LL * l * l * l_sum - static_cast<long long>(N_GROUPS) * l * l * l;
        exp_deltas[i] = reward_base * calculate_mmr_factor(static_cast<int>(variance));
    }
    
    // Phase 3: Update Cache hierarchies
    for (int i = 0; i < N_GROUPS; i++) {
        if (lookup_hits[i]) {
            cache_hit(cache_keys[i], exp_deltas[i]);
        } else {
            vector<float>* vec_ptr = new vector<float>(embeddings[i]);
            cache_insert(cache_keys[i], vec_ptr, exp_deltas[i]);
        }
    }
    
    // Phase 4: Adaptive tuning
    if (recent_evictions > aging_threshold) {
        age_factor++;
        mmr_rate = min(0.3, mmr_rate + 0.01);
        recent_evictions = 0;
    }
    
    if (global_clock % tuning_frequency == 0) {
        fine_tune_parameters();
    }
    
    // Lattice latency recording
    clock_gettime(CLOCK_MONOTONIC, &end_ts);
    double latency = (end_ts.tv_sec - start_ts.tv_sec) * 1000.0 + (end_ts.tv_nsec - start_ts.tv_nsec) / 1000000.0;
    
    if (!is_warmup) {
        hit_latency_stats[group_hit_total].push_back(latency);
    }
}