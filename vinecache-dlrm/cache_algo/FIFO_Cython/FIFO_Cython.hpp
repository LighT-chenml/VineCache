
#ifndef FIFO_H_INCLUDED
#define FIFO_H_INCLUDED

#include <iostream>
#include <string>
#include <list>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <ctime>
#include <unistd.h>
#include <thread>
#include <fcntl.h>

using namespace std;

struct Cache_data
{
    Cache_data(vector<float> ev = vector<float>(0))
    {
        this->embedding_value = ev;
    }
    vector<float> embedding_value;
};

void init(int capacity);
void request_to_fifo(vector<int> &group_keys, int &group_total_hit, int &group_perfect_hit, vector<vector<float>> &arr_emb_weights, bool is_warmup, bool use_gpu);
void load_ev_tables();
void close_ev_tables();

#endif

// python3 setup_FIFO_Cython.py build_ext --inplace