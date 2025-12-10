#include "VineCache_Cython.hpp"

using namespace std;

// Configuration constants
const int N_GROUPS = 26;
const int N_LEVELS = 30;
const int EMBEDDING_SIZE = 36;
const int BYTES_PER_EMBEDDING = 144; // 36 floats * 4 bytes

// Q-Learning constants
const int FUNC_BUCKETS = 2;
const int MMR_BUCKETS = 10;
const int AGING_BUCKETS = 10;
const int SKIP_BUCKETS = 10;
const int LEVEL_BUCKETS = 30;
const int STATE_SPACE_SIZE = FUNC_BUCKETS * MMR_BUCKETS * AGING_BUCKETS * SKIP_BUCKETS * LEVEL_BUCKETS;
const int ACTION_SPACE_SIZE = 54;

// Cache configuration
int cache_capacity = -1;
double priority_rates[3] = {0.4, 0.8, 0.95}; // p_rate1, p_rate2, p_rate3
int priority_thresholds[3]; // p1, p2, p3

// Cache state
int current_size = 0;
list<Cache_data> level_lists[N_LEVELS];
unordered_map<string, list<Cache_data>::iterator> cache_index;

// Algorithm parameters
int timestamp = 0;
int eviction_count = 0;
int age = 100;
int exp_function_type = 0;
double mmr_rate = 0.3;
int aging_threshold;
int skipping_threshold = 50;

// Performance tracking
int total_group_hits = 0;
int last_total_group_hits = 0;
int last_period_group_hits = 0;
int total_level_sum = 0;
vector<double> lookup_times[N_GROUPS + 1];
vector<int> file_descriptors(N_GROUPS);
string EV_TABLE_PATH = "/mnt/workssd/xxx/stored_model/criteo_kaggle_all_mmap/epoch-00/ev-table/binary/";

// Q-Learning state
double q_table[STATE_SPACE_SIZE][ACTION_SPACE_SIZE];
int current_state = -1;
int current_action = -1;
int fine_tune_frequency = 10000;
double learning_rate = 0.1;
double discount_factor = 0.99;
double exploration_rate = 0.1;
double min_exploration_rate = 0.01;
double exploration_decay = 0.001;

// Parameter adjustment steps
double mmr_step;
int aging_step;
int skip_step;

/**
 * Initialize VineCache cache with given capacity
 */
void init(int capacity) {
    srand(19);
    
    // Cache configuration
    cache_capacity = capacity;
    aging_threshold = cache_capacity * 0.1;
    
    // Initialize level lists
    for (int i = 0; i < N_LEVELS; i++) {
        level_lists[i].clear();
    }
    cache_index.clear();
    
    // Initialize priority thresholds
    priority_thresholds[0] = priority_rates[0] * N_GROUPS;
    priority_thresholds[1] = priority_rates[1] * N_GROUPS;
    priority_thresholds[2] = priority_rates[2] * N_GROUPS;
    
    // Initialize parameter adjustment steps
    mmr_step = mmr_rate * 0.05;
    aging_step = aging_threshold * 0.05;
    skip_step = skipping_threshold * 0.05;
    
    // Reset counters
    current_size = 0;
    timestamp = 0;
    eviction_count = 0;
    total_group_hits = 0;
    last_total_group_hits = 0;
    last_period_group_hits = 0;
    total_level_sum = 0;
    
    // Initialize Q-learning
    memset(q_table, 0, sizeof(q_table));
    current_state = -1;
    current_action = -1;
}

/**
 * Load embedding table files
 */
void load_ev_tables() {
    for (int i = 0; i < N_GROUPS; i++) {
        string table_path = EV_TABLE_PATH + "ev-table-" + to_string(i + 1) + ".bin";
        printf("Loading: %s\n", table_path.c_str());
        
        int fd = open(table_path.c_str(), O_RDONLY | O_DIRECT);
        if (fd != -1) {
            file_descriptors[i] = fd;
        } else {
            printf("Failed to open: %s\n", table_path.c_str());
        }
    }
}

/**
 * Close embedding table files and print performance statistics
 */
void close_ev_tables() {
    // Close file descriptors
    for (int i = 0; i < N_GROUPS; i++) {
        if (file_descriptors[i] != -1) {
            close(file_descriptors[i]);
            file_descriptors[i] = -1;
        }
    }
    
    // Calculate and print performance statistics
    double total_time = 0;
    int total_requests = 0;
    
    for (int i = 0; i <= N_GROUPS; i++) {
        if (!lookup_times[i].empty()) {
            double avg_time = 0;
            for (double time : lookup_times[i]) {
                avg_time += time;
                total_time += time;
            }
            avg_time /= lookup_times[i].size();
            total_requests += lookup_times[i].size();
            
            printf("Hit count: %d, Requests: %zu, Avg time: %.6lf ms\n", 
                   i, lookup_times[i].size(), avg_time);
        }
    }
    
    if (total_requests > 0) {
        printf("Overall average: %d requests, %.6lf ms\n", 
               total_requests, total_time / total_requests);
    }
}

/**
 * Read embedding vector from file
 */
vector<float> read_embedding_from_file(int table_id, int row_id) {
    vector<float> embedding(EMBEDDING_SIZE);
    int fd = file_descriptors[table_id - 1];
    
    if (fd == -1) {
        cout << "ERROR: Invalid file descriptor for table " << table_id << endl;
        return embedding;
    }
    
    char buffer[BYTES_PER_EMBEDDING];
    lseek(fd, BYTES_PER_EMBEDDING * row_id, SEEK_SET);
    
    if (read(fd, buffer, sizeof(buffer)) != sizeof(buffer)) {
        cout << "ERROR: Failed to read embedding data" << endl;
        return embedding;
    }
    
    // Convert bytes to float array
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
        embedding[i] = *(float*)(buffer + 4 * i);
    }
    
    return embedding;
}

/**
 * Calculate level based on experience value
 */
int calculate_level(double experience) {
    int level = static_cast<int>(experience);
    int bit_count = 0;
    while (level > 0) {
        level >>= 1;
        bit_count++;
    }
    return max(0, bit_count - 2);
}

/**
 * Calculate base experience value based on hit count
 */
double calculate_base_experience(int hit_count) {
    if (exp_function_type == 0) {
        // Piecewise linear function
        if (hit_count < priority_thresholds[0]) return 0;
        if (hit_count < priority_thresholds[1]) return hit_count - priority_thresholds[0];
        if (hit_count < priority_thresholds[2]) {
            return priority_thresholds[1] - priority_thresholds[0] + 
                   (hit_count - priority_thresholds[1]) * 2;
        }
        return priority_thresholds[1] - priority_thresholds[0] + 
               (priority_thresholds[2] - priority_thresholds[1]) * 2 + 
               (hit_count - priority_thresholds[2]) * 8;
    } else {
        // Exponential function
        return exp((hit_count - 10) / 2.0);
    }
}

/**
 * Sigmoid activation function
 */
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Calculate MMR (Miss Ratio Reduction) factor
 */
double calculate_mmr(int level_variance) {
    double mmr = 0;
    if (level_variance > 0) {
        mmr = log(static_cast<double>(level_variance));
    } else if (level_variance < 0) {
        mmr = -log(static_cast<double>(-level_variance));
    }
    return sigmoid(mmr * mmr_rate);
}

/**
 * Get current state for Q-learning
 */
int get_current_state() {
    int func_bucket = exp_function_type;
    int mmr_bucket = static_cast<int>(mmr_rate / 0.1);
    int aging_bucket = static_cast<int>(aging_threshold / (cache_capacity * 0.01));
    int skip_bucket = static_cast<int>(skipping_threshold / 10);
    int level_bucket = cache_index.empty() ? 0 : static_cast<int>(total_level_sum / cache_index.size());
    
    return func_bucket * (MMR_BUCKETS * AGING_BUCKETS * SKIP_BUCKETS * LEVEL_BUCKETS) +
           mmr_bucket * (AGING_BUCKETS * SKIP_BUCKETS * LEVEL_BUCKETS) +
           aging_bucket * (SKIP_BUCKETS * LEVEL_BUCKETS) +
           skip_bucket * LEVEL_BUCKETS +
           level_bucket;
}

/**
 * Get best action for given state using Q-table
 */
int get_best_action(int state) {
    int best_action = 0;
    double best_value = q_table[state][0];
    
    for (int i = 1; i < ACTION_SPACE_SIZE; i++) {
        if (q_table[state][i] > best_value) {
            best_value = q_table[state][i];
            best_action = i;
        }
    }
    
    return best_action;
}

/**
 * Apply action to adjust algorithm parameters
 */
void apply_action(int action) {
    int func_action = action / 27;
    int mmr_action = (action / 9) % 3;
    int aging_action = (action / 3) % 3;
    int skip_action = action % 3;
    
    // Toggle function type
    if (func_action == 1) {
        exp_function_type ^= 1;
    }
    
    // Adjust MMR rate
    if (mmr_action == 1) {
        mmr_rate = max(0.1, mmr_rate - mmr_step);
    } else if (mmr_action == 2) {
        mmr_rate = min(1.0, mmr_rate + mmr_step);
    }
    
    // Adjust aging threshold
    if (aging_action == 1) {
        aging_threshold = max(0, aging_threshold - aging_step);
    } else if (aging_action == 2) {
        aging_threshold = min(static_cast<int>(cache_capacity * 0.1), aging_threshold + aging_step);
    }
    
    // Adjust skipping threshold
    if (skip_action == 1) {
        skipping_threshold = max(0, skipping_threshold - skip_step);
    } else if (skip_action == 2) {
        skipping_threshold = min(100, skipping_threshold + skip_step);
    }
}

/**
 * Select action using ε-greedy strategy
 */
int select_action(int state) {
    if (static_cast<double>(rand()) / RAND_MAX < exploration_rate) {
        return rand() % ACTION_SPACE_SIZE;
    } else {
        return get_best_action(state);
    }
}

/**
 * Calculate reward for Q-learning
 */
double calculate_reward() {
    int current_hits = total_group_hits - last_total_group_hits;
    int reward = current_hits - last_period_group_hits;
    last_total_group_hits = total_group_hits;
    last_period_group_hits = current_hits;
    return reward;
}

/**
 * Fine-tune parameters using Q-learning
 */
void fine_tune_parameters() {
    int new_state = get_current_state();
    double reward = calculate_reward();
    
    // Q-learning update
    if (current_state != -1) {
        int best_next_action = get_best_action(new_state);
        q_table[current_state][current_action] += 
            learning_rate * (reward + discount_factor * q_table[new_state][best_next_action] - 
                           q_table[current_state][current_action]);
    }
    
    // Select and apply next action
    int action = select_action(new_state);
    apply_action(action);
    
    current_state = new_state;
    current_action = action;
    
    // Decay exploration rate
    exploration_rate = max(min_exploration_rate, exploration_rate - exploration_decay);
}

/**
 * Evict entry from cache using VineCache policy
 */
void evict_entry() {
    eviction_count++;
    
    int target_level = -1;
    int oldest_timestamp = INT_MAX;
    
    // Find the level with oldest entry, considering skipping probability
    for (int level = 0; level < N_LEVELS; level++) {
        if (level_lists[level].empty()) continue;
        
        int timestamp = level_lists[level].back().last_access_timestamp;
        if (timestamp < oldest_timestamp) {
            oldest_timestamp = timestamp;
            target_level = level;
        }
        
        // Apply skipping probability
        int random_value = rand() % 1000;
        if (random_value < skipping_threshold) continue;
        
        break;
    }
    
    // Remove entry from target level
    if (target_level != -1) {
        string evicted_key = level_lists[target_level].back().key;
        level_lists[target_level].pop_back();
        cache_index.erase(evicted_key);
        current_size--;
        total_level_sum -= target_level;
    }
}

/**
 * Insert new entry into cache
 */
void cache_insert(const string& key, vector<float>* value, double experience) {
    // Evict if cache is full
    if (current_size >= cache_capacity) {
        evict_entry();
    }
    
    // Insert into appropriate level
    int level = calculate_level(experience);
    level_lists[level].push_front(Cache_data(key, value, experience, timestamp));
    cache_index[key] = level_lists[level].begin();
    current_size++;
    total_level_sum += level;
}

/**
 * Update existing cache entry on hit
 */
void cache_hit(const string& key, double experience) {
    auto iter = cache_index.find(key);
    if (iter == cache_index.end()) return;
    
    double old_experience = iter->second->exp;
    double new_experience = old_experience + experience;
    int old_level = calculate_level(old_experience);
    int new_level = calculate_level(new_experience);
    
    if (old_level != new_level) {
        // Move to new level
        level_lists[new_level].push_front(
            Cache_data(key, iter->second->embedding_value, new_experience, timestamp));
        level_lists[old_level].erase(iter->second);
        cache_index[key] = level_lists[new_level].begin();
        total_level_sum += new_level - old_level;
    } else {
        // Update within same level
        iter->second->exp = new_experience;
        iter->second->last_access_timestamp = timestamp;
        level_lists[old_level].splice(level_lists[old_level].begin(), 
                                     level_lists[old_level], iter->second);
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
void request_to_vinecache(vector<int>& group_keys, int& group_total_hit, int& group_perfect_hit, 
                       vector<vector<float>>& embeddings, bool is_warmup, bool use_gpu) {
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    vector<double> experience_deltas(N_GROUPS);
    vector<int> levels(N_GROUPS);
    vector<bool> cache_hits(N_GROUPS, false);
    vector<string> cache_keys(N_GROUPS);
    
    timestamp++;
    int hit_count = 0;
    int level_sum = 0, level_sum2 = 0, level_sum3 = 0;
    
    // Phase 1: Check cache and collect data
    for (int i = 0; i < N_GROUPS; i++) {
        string key = generate_cache_key(i + 1, group_keys[i]);
        cache_keys[i] = key;
        
        auto iter = cache_index.find(key);
        if (iter != cache_index.end()) {
            // Cache hit
            embeddings[i] = *(iter->second->embedding_value);
            cache_hits[i] = true;
            hit_count++;
            levels[i] = calculate_level(iter->second->exp);
            
            // Accumulate level statistics for MMR calculation
            level_sum += levels[i];
            level_sum2 += levels[i] * levels[i];
            level_sum3 += levels[i] * levels[i] * levels[i];
        } else {
            // Cache miss
            embeddings[i] = read_embedding_from_file(i + 1, group_keys[i]);
            cache_hits[i] = false;
            group_perfect_hit = 0;
        }
    }
    
    // Update global hit statistics
    if (hit_count == N_GROUPS) {
        total_group_hits++;
    }
    group_total_hit = hit_count;
    
    // Phase 2: Calculate experience deltas
    double base_experience = calculate_base_experience(hit_count) * age / 100.0;
    for (int i = 0; i < N_GROUPS; i++) {
        int level = levels[i];
        int variance = level_sum3 - 3 * level * level_sum2 + 
                      3 * level * level * level_sum - N_GROUPS * level * level * level;
        experience_deltas[i] = base_experience * calculate_mmr(variance);
    }
    
    // Phase 3: Update cache
    for (int i = 0; i < N_GROUPS; i++) {
        if (cache_hits[i]) {
            cache_hit(cache_keys[i], experience_deltas[i]);
        } else {
            vector<float>* embedding_ptr = new vector<float>(embeddings[i]);
            cache_insert(cache_keys[i], embedding_ptr, experience_deltas[i]);
        }
    }
    
    // Adaptive parameter adjustment
    if (eviction_count > aging_threshold) {
        age++;
        mmr_rate = min(0.3, mmr_rate + 0.01);
        eviction_count = 0;
    }
    
    // Q-learning parameter tuning
    if (timestamp % fine_tune_frequency == 0) {
        fine_tune_parameters();
    }
    
    // Record performance metrics
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    if (!is_warmup) {
        lookup_times[group_total_hit].push_back(elapsed_ms);
    }
}