
#ifndef EVLFU_H_INCLUDED
#define EVLFU_H_INCLUDED

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
    Cache_data(vector<float> ev = vector<float>(0), int agg_hit = 0)
    {
        this->embedding_value = ev;
        this->agg_hit = agg_hit;
    }
    vector<float> embedding_value;
    int agg_hit;
};

void init(int capacity);
void request_to_ev_lfu(vector<int> &group_keys, int &group_total_hit, int &group_perfect_hit, vector<vector<float>> &arr_emb_weights, bool is_warmup, bool use_gpu);
void load_ev_tables();
void close_ev_tables();

#endif