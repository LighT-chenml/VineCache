
#ifndef VineCache_MFU_H_INCLUDED
#define VineCache_MFU_H_INCLUDED

#include <iostream>
#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <ctime>
#include <unistd.h>
#include <thread>
#include <cmath>
#include <set>
#include <fcntl.h>

using namespace std;

struct Cache_data
{
    Cache_data(string k = "", vector<float> *ev = new vector<float>(), double e = 0, int c = 0, int t = 0)
    {
        this->key = k;
        this->embedding_value = ev;
        this->exp = e;
        this->cnt = c;
        this->last_access_timestamp = t;
    }
    int cnt;
    int last_access_timestamp;
    double exp;
    string key;
    vector<float> *embedding_value;
    bool operator < (const Cache_data &t) const 
    {
        return cnt > t.cnt || cnt == t.cnt && key < t.key;
    }
};

void init(int capacity);
void request_to_vinecache(vector<int> &group_keys, int &group_total_hit, int &group_perfect_hit, vector<vector<float>> &arr_emb_weights, bool is_warmup, bool use_gpu);
void load_ev_tables();
void close_ev_tables();

#endif

// python3 setup_VineCache_Cython.py build_ext --inplace