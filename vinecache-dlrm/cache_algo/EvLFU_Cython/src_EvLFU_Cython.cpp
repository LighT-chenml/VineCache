#include "EvLFU_Cython.hpp"

using namespace std;

// Configuration constants
const int N_GROUPS = 26;
const int EMBEDDING_SIZE = 36;
const int BYTES_PER_EMBEDDING = 144; // 36 floats * 4 bytes

// Cache state variables
int cache_capacity = -1;
int min_frequency = 0;
unordered_map<string, Cache_data> cache_entries;
unordered_map<int, list<string>> frequency_lists;

// Perfect hit tracking and flushing parameters
int perfect_hit_count = 0;
int max_perfect_hits = 0;
double flush_rate = 0.1;
double perfect_hit_ratio = 0.85;

// Performance tracking
vector<double> lookup_times[N_GROUPS + 1];
vector<int> file_descriptors(N_GROUPS);
string EV_TABLE_PATH = "/mnt/workssd/xxx/stored_model/criteo_kaggle_all_mmap/epoch-00/ev-table/binary/";
string WORKLOAD_PATH = "/home/cc/workload/Archive-new-0.5M/";

/**
 * Utility function to split strings by delimiter
 */
vector<string> split(const string& text, const string& delimiter) {
    vector<string> tokens;
    size_t pos = 0;
    size_t len = text.length();
    size_t delim_len = delimiter.length();
    
    if (delim_len == 0) return tokens;
    
    while (pos < len) {
        size_t found = text.find(delimiter, pos);
        if (found == string::npos) {
            tokens.push_back(text.substr(pos));
            break;
        }
        tokens.push_back(text.substr(pos, found - pos));
        pos = found + delim_len;
    }
    return tokens;
}

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
 * Initialize EvLFU cache with given capacity
 */
void init(int capacity) {
    cache_capacity = capacity;
    min_frequency = 0;
    cache_entries.clear();
    
    // Initialize frequency lists for all possible frequencies
    for (int i = 0; i <= N_GROUPS; i++) {
        frequency_lists[i] = list<string>();
    }
    
    max_perfect_hits = static_cast<int>(cache_capacity * perfect_hit_ratio);
}

/**
 * Flush cache entries when perfect hit threshold is exceeded
 */
void flush_cache() {
    printf("Flushing cache!\n");
    printf("Perfect hit count: %d, Max allowed: %d\n", perfect_hit_count, max_perfect_hits);
    
    int entries_to_flush = static_cast<int>(flush_rate * cache_capacity);
    
    for (int i = 0; i < entries_to_flush && !frequency_lists[N_GROUPS].empty(); i++) {
        string key_to_evict = frequency_lists[N_GROUPS].front();
        frequency_lists[N_GROUPS].pop_front();
        cache_entries.erase(key_to_evict);
    }
    
    perfect_hit_count -= entries_to_flush;
}

/**
 * Evict least frequently used entry to make space
 */
void evict_lfu_entry() {
    // Find the minimum frequency list that has entries
    while (frequency_lists[min_frequency].empty()) {
        min_frequency++;
        if (min_frequency > N_GROUPS) {
            min_frequency = 1;
        }
    }
    
    string key_to_evict = frequency_lists[min_frequency].front();
    frequency_lists[min_frequency].pop_front();
    cache_entries.erase(key_to_evict);
}

/**
 * Add new entry to cache with given frequency
 */
void cache_put(const string& key, const vector<float>& value, int frequency) {
    // Handle cache overflow
    if (perfect_hit_count >= max_perfect_hits) {
        flush_cache();
    } else if (static_cast<int>(cache_entries.size()) >= cache_capacity) {
        evict_lfu_entry();
    }
    
    // Insert new entry
    cache_entries[key] = Cache_data(value, frequency);
    frequency_lists[frequency].push_back(key);
    
    // Update minimum frequency if necessary
    if (frequency < min_frequency) {
        min_frequency = frequency;
    }
}

/**
 * Update frequency of existing cache entry
 */
vector<float> update_entry_frequency(const string& key, int new_frequency) {
    auto iter = cache_entries.find(key);
    if (iter == cache_entries.end()) {
        return vector<float>(); // Cache miss
    }
    
    Cache_data& entry = iter->second;
    
    // Update frequency if new frequency is higher
    if (entry.agg_hit < new_frequency) {
        frequency_lists[entry.agg_hit].remove(key);
        frequency_lists[new_frequency].push_back(key);
        entry.agg_hit = new_frequency;
    }
    
    return entry.embedding_value;
}

/**
 * Read embedding vector from file for given table and row
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
 * Get or load embedding with frequency-based caching
 */
vector<float> get_or_load_embedding(const string& key, int table_id, int row_id, 
                                   int frequency, const vector<float>& preloaded_value = vector<float>()) {
    // Try to update existing entry
    vector<float> cached_value = update_entry_frequency(key, frequency);
    if (!cached_value.empty()) {
        return cached_value;
    }
    
    // Load from file if not preloaded
    vector<float> embedding = preloaded_value.empty() ? 
                             read_embedding_from_file(table_id, row_id) : preloaded_value;
    
    // Add to cache
    cache_put(key, embedding, frequency);
    return embedding;
}

/**
 * Generate cache key for table and row
 */
string generate_cache_key(int table_id, int row_id) {
    return to_string(table_id) + "-" + to_string(row_id);
}

/**
 * Process cache requests for a batch of embedding lookups using EvLFU algorithm
 */
void request_to_ev_lfu(vector<int>& group_keys, int& total_hits, int& perfect_hit, 
                       vector<vector<float>>& embeddings, bool is_warmup, bool use_gpu) {
    
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    
    vector<bool> cache_hits(N_GROUPS, false);
    vector<string> cache_keys(N_GROUPS);
    int hit_count = 0;
    perfect_hit = 1; // Assume perfect hit initially
    
    // Phase 1: Check cache status and prepare keys
    for (int i = 0; i < N_GROUPS; i++) {
        int row_id = group_keys[i];
        string key = generate_cache_key(i + 1, row_id);
        cache_keys[i] = key;
        
        if (cache_entries.count(key) > 0) {
            cache_hits[i] = true;
            hit_count++;
        } else {
            cache_hits[i] = false;
            perfect_hit = 0;
        }
    }
    
    total_hits = hit_count;
    
    // Phase 2: Preload all missing embeddings from storage
    vector<vector<float>> preloaded_embeddings(N_GROUPS);
    for (int i = 0; i < N_GROUPS; i++) {
        preloaded_embeddings[i] = read_embedding_from_file(i + 1, group_keys[i]);
    }
    
    // Phase 3: Process each embedding request
    for (int i = 0; i < N_GROUPS; i++) {
        if (cache_hits[i]) {
            // Cache hit - update frequency
            embeddings[i] = get_or_load_embedding(cache_keys[i], i + 1, group_keys[i], hit_count);
        } else {
            // Cache miss - load with preloaded data
            embeddings[i] = get_or_load_embedding(cache_keys[i], i + 1, group_keys[i], 
                                                hit_count, preloaded_embeddings[i]);
        }
    }
    
    // Update perfect hit tracking
    if (hit_count == N_GROUPS) {
        perfect_hit_count = frequency_lists[N_GROUPS].size();
    }
    
    // Record performance metrics
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double elapsed_ms = (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
                       (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
    
    if (!is_warmup) {
        lookup_times[total_hits].push_back(elapsed_ms);
    }
}

// int main()
// {
//     // The following [1] code is for algo consistency testing.
//     // The uncommented code [2] generate the workload randomly (for quick test).

//     //[1] For real workload test.
//     string workload_dir = WORKLOAD_PATH;
//     string *workload_files = new string[n_group];
//     vector<vector<int>> arrRawWorkload;
//     for (int i = 1; i <= n_group; i++)
//     {
//         workload_files[i - 1] = workload_dir;
//         workload_files[i - 1] += "workload-group-" + to_string(i);
//         workload_files[i - 1] += ".csv";
//     }

//     for (int i = 0; i < n_group; i++)
//     {
//         ifstream in(workload_files[i]);
//         string line;
//         vector<int> workload;
//         if (in.fail())
//         {
//             cout << "File not found" << endl;
//             return 0;
//         }
//         while (getline(in, line) && in.good())
//         {
//             workload.push_back(atoi(split(line, "-")[1].c_str()));
//         }
//         in.close();
//         // cout << "end!" << endl;
//         arrRawWorkload.push_back(workload);
//     }
//     vector<vector<int>> groupedWorkloadKeys = vector<vector<int>>(arrRawWorkload[0].size());
//     for (int i = 0; i < arrRawWorkload[0].size(); i++)
//     {
//         vector<int> group_keys = vector<int>(n_group);
//         for (int j = 0; j < arrRawWorkload.size(); j++)
//         {
//             group_keys[j] = arrRawWorkload[j][i];
//         }
//         groupedWorkloadKeys[i] = group_keys;
//     }
//     //// [2] for random quick test:
//     // int totalWorkload = 1000000; // 1 million
//     // vector<vector<int>> groupedWorkloadKeys(totalWorkload, vector<int>(n_group, 0));
//     // for (int i = 0; i < totalWorkload; i++)
//     //{
//     //     for (int j = 0; j < n_group; j++)
//     //     {
//     //         groupedWorkloadKeys[i][j] = rand();
//     //     }
//     // }

//     // Done merging ALL workloads!
//     // Run the alg:
//     load_ev_tables();
//     init(768);
//     int perfectHit = 0;
//     cout << "Start caching!!" << endl;
//     clock_t startTime, endTime;
//     startTime = clock();
//     for (int i = 0; i < groupedWorkloadKeys.size(); i++)
//     {
//         vector<bool> aggHitMissRecord = vector<bool>(n_group);
//         vector<vector<float>> emb_weights = vector<vector<float>>(n_group);
//         request_to_ev_lfu(groupedWorkloadKeys[i], aggHitMissRecord, emb_weights);
//         // for (int x = 0; x < n_group; x++) {
//         //     for (int y = 0; y < 36; y++) {
//         //         cout << emb_weights[x][y] << " ";
//         //     }
//         //     cout << endl;
//         // }
//         bool flag = true;
//         for (int j = 0; j < aggHitMissRecord.size(); j++)
//         {
//             if (!aggHitMissRecord[j])
//             {
//                 flag = false;
//                 break;
//             }
//         }
//         if (flag)
//         {
//             perfectHit++;
//         }
//     }
//     printf("perfect hit:%d\n", perfectHit);
//     endTime = clock();
//     cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
//     close_ev_tables();
//     return 0;
// }