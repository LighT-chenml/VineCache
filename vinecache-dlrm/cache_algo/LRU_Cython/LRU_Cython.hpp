
#ifndef LRU_H_INCLUDED
#define LRU_H_INCLUDED

#include <iostream>
#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <ctime>
#include <unistd.h>
#include <fcntl.h>

using namespace std;

struct Cache_data
{
    Cache_data(string k = "", vector<float> ev = vector<float>(0))
    {
        this->key = k;
        this->embedding_value = ev;
    }
    string key;
    vector<float> embedding_value;
};

void init(int capacity);
void request_to_lru(vector<int> &group_keys, int &group_total_hit, int &group_perfect_hit, vector<vector<float>> &arr_emb_weights, bool is_warmup, bool use_gpu);
void load_ev_tables();
void close_ev_tables();

#endif

// python3 setup_LRU_Cython.py build_ext --inplace