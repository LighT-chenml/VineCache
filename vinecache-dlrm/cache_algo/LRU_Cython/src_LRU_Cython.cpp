#include "LRU_Cython.hpp"

using namespace std;

// Configuration constants
const int N_GROUPS = 26;
const int EMBEDDING_SIZE = 36;
const int BYTES_PER_EMBEDDING = 144; // 36 floats * 4 bytes

// Cache state
int cache_capacity = -1;
list<Cache_data> lru_list;
unordered_map<string, list<Cache_data>::iterator> cache_index;

// Performance tracking
vector<double> lookup_times[N_GROUPS + 1];
vector<int> file_descriptors(N_GROUPS);
string EV_TABLE_PATH = "/mnt/workssd/xxx/stored_model/criteo_kaggle_all_mmap/epoch-00/ev-table/binary/";

/**
 * Initialize file descriptors for all embedding tables
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
 * Close all file descriptors and print performance statistics
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
 * Read embedding vector from file for given table and row
 */
vector<float> read_embedding_from_file(int table_id, int row_id) {
    vector<float> embedding(EMBEDDING_SIZE);
    int fd = file_descriptors[table_id - 1];
    
    if (fd == -1) {
        cout << "ERROR: Invalid file descriptor for table " << table_id << endl;
        return embedding; // Return zero-initialized vector
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
 * Initialize LRU cache with given capacity
 */
void init(int capacity) {
    cache_capacity = capacity;
    lru_list.clear();
    cache_index.clear();
}

/**
 * Move accessed item to front of LRU list (most recently used)
 */
void move_to_front(const string& key) {
    auto iter = cache_index.find(key);
    if (iter != cache_index.end()) {
        lru_list.splice(lru_list.begin(), lru_list, iter->second);
    }
}

/**
 * Add or update cache entry using LRU eviction policy
 */
void cache_put(const string& key, const vector<float>& value) {
    // Evict least recently used entry if cache is full
    if (static_cast<int>(lru_list.size()) >= cache_capacity) {
        string evicted_key = lru_list.back().key;
        lru_list.pop_back();
        cache_index.erase(evicted_key);
    }

    // Add new entry to front (most recently used)
    lru_list.push_front(Cache_data(key, value));
    cache_index[key] = lru_list.begin();
}

/**
 * Generate cache key for table and row
 */
string generate_cache_key(int table_id, int row_id) {
    return to_string(table_id) + "-" + to_string(row_id);
}

/**
 * Process cache requests for a batch of embedding lookups
 */
void request_to_lru(vector<int>& group_keys, int& total_hits, int& perfect_hit, 
                    vector<vector<float>>& embeddings, bool is_warmup, bool use_gpu) {
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    vector<bool> cache_hits(N_GROUPS, false);
    vector<string> cache_keys(N_GROUPS);
    int hit_count = 0;
    perfect_hit = 1; // Assume perfect hit initially

    // Phase 1: Check cache and prepare data
    for (int i = 0; i < N_GROUPS; i++) {
        string key = generate_cache_key(i + 1, group_keys[i]);
        cache_keys[i] = key;

        auto cache_iter = cache_index.find(key);
        if (cache_iter != cache_index.end()) {
            // Cache hit - get value and mark as recently used
            embeddings[i] = cache_iter->second->embedding_value;
            cache_hits[i] = true;
            hit_count++;
        } else {
            // Cache miss - read from file
            embeddings[i] = read_embedding_from_file(i + 1, group_keys[i]);
            cache_hits[i] = false;
            perfect_hit = 0;
        }
    }

    // Phase 2: Update cache based on access patterns
    for (int i = 0; i < N_GROUPS; i++) {
        if (cache_hits[i]) {
            // Move to front for LRU ordering
            move_to_front(cache_keys[i]);
        } else {
            // Add new entry to cache
            cache_put(cache_keys[i], embeddings[i]);
        }
    }

    // Record performance metrics
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;

    total_hits = hit_count;
    
    if (!is_warmup) {
        lookup_times[hit_count].push_back(elapsed_ms);
    }
}