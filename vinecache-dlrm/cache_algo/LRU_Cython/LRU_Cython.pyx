from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string

cdef extern from "LRU_Cython.hpp":
    void init(int capacity)
    void request_to_lru(vector[int] &group_keys, int &group_total_hit, int &group_perfect_hit, vector[vector[float]] &arr_emb_weights, bool is_warmup, bool use_gpu)
    void load_ev_tables()
    void close_ev_tables()

def cinit(int capacity):
    init(capacity)

def crequest(vector[int] group_keys, is_warmup, use_gpu):
    cdef int group_total_hit = 0
    cdef int group_perfect_hit = 1
    cdef vector[vector[float]] arr_emb_weights = [[0.0]*36]*26
    request_to_lru(group_keys, group_total_hit, group_perfect_hit, arr_emb_weights, is_warmup, use_gpu)
    return group_total_hit, group_perfect_hit, arr_emb_weights


def cload_ev_tables():
    load_ev_tables()

def cclose_ev_tables():
    close_ev_tables()