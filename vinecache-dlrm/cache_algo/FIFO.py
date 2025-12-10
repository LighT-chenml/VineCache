import collections
import torch
import sys
sys.path.append('emb_storage')
import storage_manager

cap = -1
FIFOCache = collections.OrderedDict()# each item is a dictionary embedding value

def init(capacity):
    global cap 
    cap = capacity

# Inserting the NEW key
def set(key, value):
    global FIFOCache, cap
    if (len(FIFOCache) >= cap):
        # evicting the first key in LRU list
        evict_key, evict_val = FIFOCache.popitem(last=False)
    # inserting new key
    FIFOCache[key] = value

# single key request
def request(key, table_id, row_id):
    global FIFOCache
    if (key in FIFOCache):
        value = FIFOCache[key]
        return value, True
    else:
        # MISS: get value from secondary storage
        value = storage_manager.get_val_from_storage(table_id, row_id)
        set(key, value)
        return value, False

# Multi keys request
def request_to_fifo( group_row_ids, use_gpu = False):
    arr_record_hit = []
    arr_emb_weights = []
    if (use_gpu):
        # This code assume that we only run this on a single GPU node
        device = torch.device("cuda:0")

    total_hit = 0
    perfect_hit = 1

    for i, row_id in enumerate(group_row_ids):
        # Table_id is started at 1
        # Key for row3 of table1 is 1-3
        key = str(i+1) + "-" + str(row_id)
        val, is_hit = request(key, i+1, row_id)
        # convert list of embedding values to tensor 
        ev_tensor = torch.FloatTensor([val]) # val is a python list
        ev_tensor.requires_grad = True
        if (use_gpu):
            # This code assume that we only run this on a single GPU node
            ev_tensor = ev_tensor.to(device)
        arr_emb_weights.append(ev_tensor)
        if is_hit:
            # arr_record_hit.append(True)
            total_hit += 1
        else:
            # arr_record_hit.append(False)
            perfect_hit = 0
    
    return total_hit, perfect_hit, arr_emb_weights

