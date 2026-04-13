"""
Deep Learning Recommendation Model (DLRM) implementation.
The model processes dense and sparse features through MLPs and embedding lookups.
Features integration with VineCache for high-performance embedding retrieval.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

# Standard library imports
import argparse
import builtins
import datetime
import json
import sys
import time
import csv
import os
import struct
import warnings
import subprocess
from pathlib import Path
from collections import OrderedDict

# Third-party imports
import pandas as pd
import numpy as np
import sklearn.metrics
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter

# Custom module paths setup
sys.path.extend([
    'emb_storage',
    'cache_algo',
    'cache_algo/EvLFU_Cython',
    'cache_algo/FIFO_Cython',
    'cache_algo/LRU_Cython',
    'cache_algo/VineCache_Cython',
    'cache_algo/VineCache_MFU_Cython'
])

# Cache and Storage imports
import EvLFU
import LRU
import LFU
import FIFO
import storage_manager
import evstore_utils
import EvLFU_Cython
import FIFO_Cython
import LRU_Cython
import VineCache_Cython
import VineCache_MFU_Cython

# DLRM specific modules
import dlrm_data_pytorch as dp
import extend_distributed as ext_dist
import optim.rwsadagrad as RowWiseSparseAdagrad
from tricks.md_embedding_bag import PrEmbeddingBag, md_solver
from tricks.qr_embedding_bag import QREmbeddingBag

# --- Global Constants ---
STORED_MODEL_PATH = "stored_model"
DEFAULT_EPOCH = "epoch-00"
TRAIN_FILE = "train.txt"
TEST_FILE = "test.txt"
AUC_METRICS_FILE = "metrics.txt"

# --- Performance Counters ---
perfect_hit = 0
total_hit = 0
key_counter = set()
request_cnt = 0


def time_wrap(use_gpu):
    """
    Synchronized time measurement for GPU/CPU.
    """
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, is_warmup, ndevices=1):
    """
    Wrapper for DLRM forward pass with device handling and activity recording.
    """
    with record_function("DLRM forward"):
        if use_gpu and ndevices == 1:
            # Transfer sparse features to the target device
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) else lS_o.to(device)
        return dlrm(X.to(device), lS_o, lS_i, is_warmup)


def unpack_batch(batch):
    """
    Helper to unpack batch components for processing.
    """
    X, lS_o, lS_i, targets = batch[0], batch[1], batch[2], batch[3]
    return X, lS_o, lS_i, targets, torch.ones(targets.size()), None


def apply_emb_ori_dlrm(lS_o, lS_i, emb_l, v_W_l):
    """Original DLRM embedding lookup implementation"""
    ly = []
    for k, sparse_index_group_batch in enumerate(lS_i):
        sparse_offset_group_batch = lS_o[k]
        
        # Handle per-sample weights
        per_sample_weights = (
            v_W_l[k].gather(0, sparse_index_group_batch) 
            if v_W_l[k] is not None else None
        )
        
        # Embedding lookup using EmbeddingBag
        E = emb_l[k]
        V = E(sparse_index_group_batch, sparse_offset_group_batch, 
              per_sample_weights=per_sample_weights)
        ly.append(V)
    
    return ly


def apply_emb_evstore(lS_o, lS_i, emb_l, v_W_l, is_warmup, use_gpu=False, 
                      use_emb_cache=False, approx_emb_threshold=-1):
    """
    Retrieves embeddings using EVStore with optional caching support.
    Dispatches requests to various cache algorithms.
    Handles both single-feature and multi-feature batches.
    """
    global cache_algo
    
    device = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    
    # Standardize lS_i into a list of numpy arrays (one per feature)
    if isinstance(lS_i, list):
        lS_i_list = [S_i.cpu().detach().numpy() for S_i in lS_i]
    else:
        # If it's a tensor of shape (batch_size, num_features)
        lS_i_cpu = lS_i.cpu().detach().numpy()
        if len(lS_i_cpu.shape) == 1:
            lS_i_list = [lS_i_cpu]
        else:
            lS_i_list = [lS_i_cpu[:, i] for i in range(lS_i_cpu.shape[1])]
    
    batch_size = len(lS_i_list[0])
    num_features = len(lS_i_list)
    
    all_ly = []
    g_total_hit = 0
    g_perfect_hit = 0
    
    # Process each sample in the batch
    # Note: Optimization might be needed for large batches, 
    # but this ensures correctness given the current EVStore interface.
    for b in range(batch_size):
        group_row_ids = [int(lS_i_list[f][b]) for f in range(num_features)]
        
        if use_emb_cache:
            # Dispatch table for cache implementations
            cache_handlers = {
                "evlfu":                lambda: EvLFU.request_to_ev_lfu(group_row_ids, use_gpu, approx_emb_threshold),
                "lru":                  lambda: LRU.request_to_lru(group_row_ids, use_gpu),
                "lfu":                  lambda: LFU.request_to_lfu(group_row_ids, use_gpu),
                "fifo":                 lambda: FIFO.request_to_fifo(group_row_ids, use_gpu),
                "evlfu_cython":         lambda: EvLFU_Cython.crequest(group_row_ids, is_warmup, use_gpu),
                "fifo_cython":          lambda: FIFO_Cython.crequest(group_row_ids, is_warmup, use_gpu),
                "lru_cython":           lambda: LRU_Cython.crequest(group_row_ids, is_warmup, use_gpu),
                "vinecache_cython":     lambda: VineCache_Cython.crequest(group_row_ids, is_warmup, use_gpu),
                "vinecache_mfu_cython": lambda: VineCache_MFU_Cython.crequest(group_row_ids, is_warmup, use_gpu)
            }
            
            if cache_algo in cache_handlers:
                resp = cache_handlers[cache_algo]()
                
                if cache_algo.endswith("_cython"):
                    t_hit, p_hit, emb_data = resp
                    # emb_data is vector[vector[float]], size (num_features, m_spa)
                    sample_ly = [torch.FloatTensor(emb_data[i]).to(device) for i in range(len(emb_data))]
                else:
                    t_hit, p_hit, sample_ly = resp
                
                g_total_hit += t_hit
                g_perfect_hit += p_hit
                all_ly.append(sample_ly)
            else:
                raise ValueError(f"[DLRM] Unsupported cache algorithm: {cache_algo}")
        else:
            # Direct fallback to storage manager
            _, sample_ly = storage_manager.request_to_emb_storage(group_row_ids, use_gpu)
            all_ly.append(sample_ly)
    
    # Reconstruct ly from all_ly
    # all_ly is (batch_size, num_features) tensors of shape (1, m_spa) or (m_spa,)
    # We need ly: (num_features) list of tensors of shape (batch_size, m_spa)
    final_ly = []
    for f in range(num_features):
        feature_tensors = []
        for b in range(batch_size):
            t = all_ly[b][f]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            feature_tensors.append(t)
        final_ly.append(torch.cat(feature_tensors, dim=0))
    
    if use_emb_cache:
        return g_total_hit, g_perfect_hit, final_ly
    else:
        return final_ly


def calculate_and_write_cdf(cdf_output_dir, cache_algo, arr_latency):
    """Calculate and write CDF of latency data"""
    Path(cdf_output_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(arr_latency, columns=['latency_ms'])
    df['latency_ms'] = df['latency_ms'] * 1000  # Convert to milliseconds
    
    output_path = os.path.join(cdf_output_dir, f"{cache_algo}-cdf.csv")
    df.to_csv(output_path, sep=',', index=False)
    print(f"CDF latency data written to: {output_path}")


class DLRM_Net(nn.Module):
    """Deep Learning Recommendation Model Neural Network"""
    
    def __init__(self, m_spa=None, ln_emb=None, ln_bot=None, ln_top=None,
                 arch_interaction_op=None, arch_interaction_itself=False,
                 sigmoid_bot=-1, sigmoid_top=-1, sync_dense_params=True,
                 loss_threshold=0.0, ndevices=-1, qr_flag=False,
                 qr_operation="mult", qr_collisions=0, qr_threshold=200,
                 md_flag=False, md_threshold=200, weighted_pooling=None,
                 loss_function="bce", use_gpu=False, use_evstore=False,
                 use_emb_cache=False, inference_only=False, ev_lookup_only=False,
                 approx_emb_threshold=-1):
        
        super(DLRM_Net, self).__init__()
        
        # Validate required parameters
        required_params = [m_spa, ln_emb, ln_bot, ln_top, arch_interaction_op]
        if not all(param is not None for param in required_params):
            raise ValueError("Missing required parameters for DLRM initialization")
        
        # Store configuration
        self._init_config(locals())
        
        # Handle distributed setup
        self._setup_distributed(ln_emb)
        
        # Create model components
        if ndevices <= 1:
            self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
            self._setup_weights(w_list)
        
        self.bot_l = self.create_mlp(ln_bot, sigmoid_bot)
        self.top_l = self.create_mlp(ln_top, sigmoid_top)
        
        # Setup loss function
        self._setup_loss_function()
    
    def _init_config(self, params):
        """Initialize configuration from parameters"""
        config_map = {
            'approx_emb_threshold': 'approx_emb_threshold',
            'ev_lookup_only': 'testing_init_overhead',
            'inference_only': 'inference_only',
            'use_gpu': 'use_gpu',
            'use_evstore': 'use_evstore',
            'use_emb_cache': 'use_emb_cache',
            'ndevices': 'ndevices',
            'arch_interaction_op': 'arch_interaction_op',
            'arch_interaction_itself': 'arch_interaction_itself',
            'sync_dense_params': 'sync_dense_params',
            'loss_threshold': 'loss_threshold',
            'loss_function': 'loss_function',
            'weighted_pooling': 'weighted_pooling',
            'qr_flag': 'qr_flag',
            'md_flag': 'md_flag'
        }
        
        for param_name, attr_name in config_map.items():
            setattr(self, attr_name, params.get(param_name))
        
        # Special handling for some attributes
        self.ev_lookup_only = params.get('ev_lookup_only')
        self.output_d = 0
        self.parallel_model_batch_size = -1
        self.parallel_model_is_not_prepared = True
        
        # Handle weighted pooling
        if self.weighted_pooling is not None and self.weighted_pooling != "fixed":
            self.weighted_pooling = "learned"
        
        # Setup QR and MD flags
        if self.qr_flag:
            self.qr_collisions = params.get('qr_collisions')
            self.qr_operation = params.get('qr_operation')
            self.qr_threshold = params.get('qr_threshold')
        
        if self.md_flag:
            self.md_threshold = params.get('md_threshold')
        
        # Quantization setup
        self.quantize_emb = False
        self.emb_l_q = []
        self.quantize_bits = 32
    
    def _setup_distributed(self, ln_emb):
        """Setup distributed training configuration"""
        if ext_dist.my_size > 1:
            n_emb = len(ln_emb)
            if n_emb < ext_dist.my_size:
                sys.exit(f"Only {n_emb} sparse features for {ext_dist.my_size} devices")
            
            self.n_global_emb = n_emb
            self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(n_emb)
            self.local_emb_slice = ext_dist.get_my_slice(n_emb)
            self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]
    
    def _setup_weights(self, w_list):
        """Setup embedding weights"""
        if self.weighted_pooling == "learned":
            self.v_W_l = nn.ParameterList([Parameter(w) for w in w_list])
        else:
            self.v_W_l = w_list
    
    def _setup_loss_function(self):
        """Setup loss function based on configuration"""
        loss_functions = {
            "mse": lambda: torch.nn.MSELoss(reduction="mean"),
            "bce": lambda: torch.nn.BCELoss(reduction="mean"),
            "wbce": lambda: self._setup_weighted_bce()
        }
        
        if self.loss_function in loss_functions:
            self.loss_fn = loss_functions[self.loss_function]()
        else:
            sys.exit(f"ERROR: --loss-function={self.loss_function} is not supported")
    
    def _setup_weighted_bce(self):
        """Setup weighted BCE loss"""
        # This would need access to args, which should be passed or stored
        # For now, using a placeholder
        self.loss_ws = torch.tensor([1.0, 1.0])  # Placeholder
        return torch.nn.BCELoss(reduction="none")
    
    def create_mlp(self, layer_sizes, sigmoid_layer_idx):
        """
        Creates an MLP (Multi-Layer Perceptron) module.
        """
        layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = int(layer_sizes[i]), int(layer_sizes[i + 1])
            
            # Linear layer with deliberate weight initialization
            linear_layer = nn.Linear(n_in, n_out, bias=True)
            
            # Xavier/Kaiming-like initialization for stability
            std_dev = np.sqrt(2 / (n_out + n_in))
            W = np.random.normal(0.0, std_dev, size=(n_out, n_in)).astype(np.float32)
            b = np.random.normal(0.0, np.sqrt(1 / n_out), size=n_out).astype(np.float32)
            
            linear_layer.weight.data = torch.tensor(W, requires_grad=True)
            linear_layer.bias.data = torch.tensor(b, requires_grad=True)
            layers.append(linear_layer)
            
            # Non-linear activation
            if i == sigmoid_layer_idx:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())
        
        return torch.nn.Sequential(*layers)
    
    def create_emb(self, m_spa, ln_emb, weighted_pooling=None):
        """
        Creates a list of embedding layers (EmbeddingBag).
        """
        emb_list = nn.ModuleList()
        v_W_list = []
        
        if self.inference_only and self.use_evstore:
            print("[DLRM] Inference-only mode enabled: minimizing RAM footprint.")
        
        for i, n_emb in enumerate(ln_emb):
            if ext_dist.my_size > 1 and i not in self.local_emb_indices:
                continue
            
            # Create the actual embedding bag
            ee = self._create_single_embedding(n_emb, m_spa, i, ln_emb)
            emb_list.append(ee)
            
            # Initialize weights for pooling if requested
            if weighted_pooling is None:
                v_W_list.append(None)
            else:
                v_W_list.append(torch.ones(n_emb, dtype=torch.float32))
        
        return emb_list, v_W_list
    
    def _create_single_embedding(self, n, m, i, ln):
        """Create a single embedding layer"""
        # QR embedding
        if self.qr_flag and n > self.qr_threshold:
            return QREmbeddingBag(n, m, self.qr_collisions, 
                                operation=self.qr_operation, mode="sum", sparse=True)
        
        # MD embedding
        if self.md_flag and n > self.md_threshold:
            base = max(m)
            _m = m[i] if n > self.md_threshold else base
            EE = PrEmbeddingBag(n, _m, base)
            W = np.random.uniform(low=-np.sqrt(1/n), high=np.sqrt(1/n), 
                                size=(n, _m)).astype(np.float32)
            EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            return EE
        
        # Standard embedding
        EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
        
        if self.inference_only and self.use_evstore:
            # Minimal memory usage for inference
            EE.weight.data = torch.tensor(np.zeros([n, m]).astype(np.int8))
        else:
            # Normal initialization
            W = np.random.uniform(low=-np.sqrt(1/n), high=np.sqrt(1/n), 
                                size=(n, m)).astype(np.float32)
            EE.weight.data = torch.tensor(W, requires_grad=True)
        
        return EE
    
    def apply_mlp(self, x, layers):
        """Apply MLP layers to input"""
        return layers(x)
    
    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l, is_warmup):
        """Apply embedding lookup"""
        if self.use_evstore:
            result = apply_emb_evstore(lS_o, lS_i, emb_l, v_W_l, is_warmup,
                                     self.use_gpu, self.use_emb_cache, 
                                     self.approx_emb_threshold)
            
            if not is_warmup:
                global total_hit, perfect_hit
                if len(result) == 3:  # With cache
                    group_total_hit, group_perfect_hit, ly = result
                    total_hit += group_total_hit
                    perfect_hit += group_perfect_hit
                    return ly
                else:  # Direct storage
                    return result
            else:
                return result[-1] if len(result) == 3 else result
        else:
            return apply_emb_ori_dlrm(lS_o, lS_i, emb_l, v_W_l)
    
    def interact_features(self, x, ly):
        """Interact dense and sparse features"""
        if self.arch_interaction_op == "dot":
            return self._dot_interaction(x, ly)
        elif self.arch_interaction_op == "cat":
            return torch.cat([x] + ly, dim=1)
        else:
            sys.exit(f"ERROR: --arch-interaction-op={self.arch_interaction_op} not supported")
    
    def _dot_interaction(self, x, ly):
        """Perform dot product interaction"""
        batch_size, d = x.shape
        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        
        # Extract unique interactions
        _, ni, nj = Z.shape
        offset = 1 if self.arch_interaction_itself else 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        Zflat = Z[:, li, lj]
        
        return torch.cat([x] + [Zflat], dim=1)
    
    def forward(self, dense_x, lS_o, lS_i, is_warmup):
        """Forward pass of DLRM"""
        if self.ndevices <= 1:
            if self.testing_init_overhead:
                return None  # Only measuring initialization overhead
            return self.sequential_forward(dense_x, lS_o, lS_i, is_warmup)
        else:
            print("ERROR: Multi-device run not supported with EVStore integration")
            sys.exit(-1)
    
    def sequential_forward(self, dense_x, lS_o, lS_i, is_warmup):
        """Sequential forward pass for single device"""
        if self.ev_lookup_only:
            # Only embedding lookup, no MLP inference
            self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, is_warmup)
            return None
        
        # Full forward pass
        x = self.apply_mlp(dense_x, self.bot_l)  # Bottom MLP
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l, is_warmup)  # Embeddings
        z = self.interact_features(x, ly)  # Feature interactions
        p = self.apply_mlp(z, self.top_l)  # Top MLP
        
        # Apply loss threshold clamping
        if 0.0 < self.loss_threshold < 1.0:
            return torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        return p


def dash_separated_ints(value):
    """Parse dash-separated integers"""
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not valid dash separated ints")
    return value


def dash_separated_floats(value):
    """Parse dash-separated floats"""
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{value} is not valid dash separated floats")
    return value


def inference(args, dlrm_model, best_acc, best_auc, loader, device, use_gpu):
    """
    Executes the inference pass on the provided data loader.
    Supports cache warmup and workload tracing.
    """
    test_accu = 0
    test_samp = 0
    latencies = []
    workload_trace = []
    
    # Optional cache warmup phase
    warmup_iters = int(len(loader) / 2) if args.cache_warmup else 0
    if warmup_iters > 0:
        print(f"[DLRM] Warming up cache for {warmup_iters} iterations...")
        _run_warmup(args, dlrm_model, loader, device, use_gpu, warmup_iters)
    
    test_iters = len(loader) - warmup_iters
    print(f"[DLRM] Starting test phase: {test_iters} iterations.")
    
    # Main inference loop
    for i, batch in tqdm(enumerate(loader), total=len(loader), desc="Inference"):
        if i < warmup_iters:
            continue
        
        X, lS_o, lS_i, targets, weights, _ = unpack_batch(batch)
        
        # Skip incomplete batches in distributed mode
        if ext_dist.my_size > 1 and X.size(0) % ext_dist.my_size != 0:
            continue
        
        if args.trace_inference_workload:
            # Trace access patterns for simulation/analysis
            workload_trace.append(_get_lookup_keys(lS_i))
        else:
            # Measure inference latency and accuracy
            t_start = time.time()
            predictions = dlrm_wrap(X, lS_o, lS_i, use_gpu, device, False)
            t_end = time.time()
            
            latencies.append(t_end - t_start)
            
            if not args.ev_lookup_only:
                acc_batch, samp_batch = _compute_batch_accuracy(predictions, targets, X)
                test_accu += acc_batch
                test_samp += samp_batch
    
    if args.trace_inference_workload:
        return workload_trace
    
    # Aggregate and return metrics
    final_acc = test_accu / test_samp if test_samp > 0 else 0
    
    metrics = {
        "nepochs": args.nepochs,
        "nbatches_test": len(loader),
        "state_dict": dlrm_model.state_dict(),
        "test_acc": final_acc,
    }
    
    is_best_acc = final_acc > best_acc
    if is_best_acc:
        best_acc = final_acc
        
    print(f"[Results] Current Accuracy: {final_acc * 100:.3f}%, Best Accuracy: {best_acc * 100:.3f}%")
    
    return metrics, is_best_acc, "", None, latencies


def _run_warmup(args, model, loader, device, use_gpu, iters):
    """
    Runs a subset of batches to prime the embedding cache.
    """
    loader_iter = iter(loader)
    for _ in range(iters):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        X, lS_o, lS_i, _, _, _ = unpack_batch(batch)
        dlrm_wrap(X, lS_o, lS_i, use_gpu, device, is_warmup=True)


def _get_lookup_keys(lS_i):
    """
    Extracts formatted keys for workload tracing.
    """
    return [f"{i+1}-{row[0]}" for i, row in enumerate(lS_i.numpy())]


def _compute_batch_accuracy(predictions, targets, inputs):
    """
    Computes accuracy for a single batch, handling distributed results.
    """
    with record_function("DLRM accuracy compute"):
        if predictions.is_cuda:
            torch.cuda.synchronize()
        
        _, split_lens = ext_dist.get_split_lengths(inputs.size(0))
        if ext_dist.my_size > 1:
            predictions = ext_dist.all_gather(predictions, split_lens)
        
        S = predictions.detach().cpu().numpy()
        T = targets.detach().cpu().numpy()
        
        batch_size = T.shape[0]
        correct_count = np.sum((np.round(S, 0) == T).astype(np.uint8))
        
        return correct_count, batch_size


def create_output_folder(args, input_data_name):
    """Create output folders for storing models and embedding tables"""
    print("Creating output folders for model and embedding tables")
    for i in range(args.nepochs):
        folder_path = os.path.join(STORED_MODEL_PATH, input_data_name, 
                                  f"epoch-{i}", "ev-table")
        Path(folder_path).mkdir(parents=True, exist_ok=True)


def setup_cache_algorithms(args):
    """
    Initializes and loads embedding cache algorithms.
    """
    if not args.use_emb_cache:
        return
    
    print(f"[Cache] Initializing cache algorithms with size: {args.cache_size}")
    
    # Initialize standard Python implementations
    python_libs = [EvLFU, LRU, LFU, FIFO]
    for lib in python_libs:
        lib.init(args.cache_size)
    
    # Initialize high-performance Cython implementations
    cython_libs = [EvLFU_Cython, FIFO_Cython, LRU_Cython, VineCache_Cython, VineCache_MFU_Cython]
    for lib in cython_libs:
        lib.cinit(args.cache_size)
    
    # Pre-load embedding tables for relevant algorithms
    loader_map = {
        "evlfu_cython":         EvLFU_Cython.cload_ev_tables,
        "fifo_cython":          FIFO_Cython.cload_ev_tables,
        "lru_cython":           LRU_Cython.cload_ev_tables,
        "vinecache_cython":     VineCache_Cython.cload_ev_tables,
        "vinecache_mfu_cython": VineCache_MFU_Cython.cload_ev_tables
    }
    
    if args.cache_algo in loader_map:
        print(f"[Cache] Loading embedding tables for {args.cache_algo}...")
        loader_map[args.cache_algo]()


def setup_storage_manager(args):
    """Setup storage manager configuration"""
    storage_manager.ev_precs = args.ev_precs
    
    storage_types = {
        "dummy": storage_manager.EmbStorage.DUMMY,
        "rocksdb": storage_manager.EmbStorage.ROCKSDB,
        "filepy": storage_manager.EmbStorage.FILEPY,
        "mmapfilepy": storage_manager.EmbStorage.MMAPFILEPY,
        "sqlite": storage_manager.EmbStorage.SQLITE,
        "filec": storage_manager.EmbStorage.FILEC
    }
    
    if args.emb_stor not in storage_types:
        print(f"ERROR: Unknown embedding storage type: {args.emb_stor}")
        sys.exit(-1)
    
    storage_manager.storage_type = storage_types[args.emb_stor]
    
    if args.use_multi_socket:
        storage_manager.init_socket_client()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train Deep Learning Recommendation Model (DLRM)")
    
    # Model architecture
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument("--arch-embedding-size", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument("--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot")
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    
    # Data parameters
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument("--data-generation", type=str, default="random")
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    
    # Training parameters
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    
    # System parameters
    parser.add_argument("--use-gpu", type=str, default="False")
    parser.add_argument("--inference-only", type=str, default="False")
    
    # EVStore parameters
    parser.add_argument("--use-evstore", type=str, default="False")
    parser.add_argument("--use-emb-cache", type=str, default="False")
    parser.add_argument("--cache-algo", type=str, default="evlfu")
    parser.add_argument("--cache-size", type=int, default=768)
    parser.add_argument("--emb-stor", type=str, default="dummy")
    parser.add_argument("--ev-lookup-only", type=str, default="False")
    parser.add_argument("--cache-warmup", type=str, default="True")
    
    # File paths
    parser.add_argument("--input-data", type=str, default="./input/default-input-data")
    parser.add_argument("--ev-path", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    
    # Additional parameters
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    
    return parser.parse_args()


def main():
    """Main execution function"""
    global args, cache_algo, dlrm
    
    # Parse arguments
    args = parse_arguments()
    
    # Convert string arguments to boolean
    bool_args = ['inference_only', 'use_gpu', 'use_evstore', 'use_emb_cache', 
                'ev_lookup_only', 'cache_warmup', 'arch_interaction_itself', 'debug_mode']
    for arg in bool_args:
        val = getattr(args, arg)
        if isinstance(val, str):
            setattr(args, arg, val.lower() in ["true", "1", "yes", "t"])
    
    cache_algo = args.cache_algo.lower()
    
    # Setup components
    setup_cache_algorithms(args)
    setup_storage_manager(args)
    
    # Setup paths and data
    input_data_name = Path(args.input_data).name
    args.raw_data_file = os.path.join(args.input_data, TRAIN_FILE)
    args.processed_data_file = os.path.join(args.input_data, f"{input_data_name}.npz")
    
    create_output_folder(args, input_data_name)
    
    # Setup random seeds and device
    np.random.seed(args.numpy_rand_seed)
    torch.manual_seed(args.numpy_rand_seed)
    
    use_gpu = args.use_gpu and torch.cuda.is_available()
    if args.use_gpu and not torch.cuda.is_available():
        print("ERROR: CUDA not available but GPU requested")
        sys.exit(-1)
    
    # Initialize distributed training
    if not args.debug_mode:
        ext_dist.init_distributed(local_rank=-1, use_gpu=use_gpu, backend="")
    
    # Setup device
    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        device = torch.device("cuda", 0)
        print("Using GPU...")
    else:
        device = torch.device("cpu")
        print("Using CPU...")
    
    # Prepare data
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    
    if args.data_generation == "dataset":
        print("Loading Criteo dataset")
        test_data, test_ld = dp.make_criteo_data_and_loaders(args)
        # Load training configuration
        training_config_path = os.path.join(STORED_MODEL_PATH, input_data_name, 
                                          evstore_utils.TRAINING_CONFIG_FILE)
        _, nbatches, nbatches_test, ln_emb, m_den = evstore_utils.read_training_config(
            training_config_path)
        ln_emb = np.array(ln_emb)
        ln_bot[0] = m_den
    else:
        # Generate random data
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = dp.make_random_data_and_loader(
            args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)
    
    # Load embedding data
    if args.use_evstore:
        assert args.ev_path != "", "EV path required for EVStore"
        storage_manager.load_ev_table_into_emb_stor(args.ev_path, False)
        print("Loaded data to embedding storage")
    
    # Calculate model dimensions
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1
    
    m_den_out = ln_bot[-1]
    if args.arch_interaction_op == "dot":
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    
    arch_mlp_top_adjusted = f"{num_int}-{args.arch_mlp_top}"
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")
    
    # Create DLRM model
    ndevices = -1  # CPU only for simplicity
    
    dlrm = DLRM_Net(
        m_spa=m_spa,
        ln_emb=ln_emb,
        ln_bot=ln_bot,
        ln_top=ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        ndevices=ndevices,
        use_gpu=use_gpu,
        use_evstore=args.use_evstore,
        use_emb_cache=args.use_emb_cache,
        inference_only=args.inference_only,
        ev_lookup_only=args.ev_lookup_only
    )
    
    if use_gpu:
        dlrm = dlrm.to(device)
    
    # Load pre-trained model if specified
    if args.load_model:
        model_path = os.path.join(STORED_MODEL_PATH, input_data_name, 
                                 DEFAULT_EPOCH, args.load_model)
        print(f"Loading model from {model_path}")
        
        if use_gpu:
            ld_model = torch.load(model_path, map_location=device)
        else:
            ld_model = torch.load(model_path, map_location=torch.device("cpu"))
        
        dlrm.load_state_dict(ld_model["state_dict"])
        print(f"Model loaded - Test accuracy: {ld_model['test_acc'] * 100:.3f}%")
    
    # Run inference execution
    if args.inference_only:
        print("\n" + "="*40)
        print("          DLRM INFERENCE START          ")
        print("="*40)
        print(f"Device: {device.type.upper()}")
        print(f"EVStore: {'Enabled' if args.use_evstore else 'Disabled'}")
        print(f"Cache: {'Enabled' if args.use_emb_cache else 'Disabled'}")
        print(f"Algorithm: {cache_algo.upper()}")
        print(f"Capacity: {args.cache_size}")
        print("-"*40)
        
        t0 = time.time()
        results = inference(args, dlrm, 0, 0, test_ld, device, use_gpu)
        t1 = time.time()
        
        # Unpack results
        if args.trace_inference_workload:
            print(f"[DLRM] Workload trace generated with {len(results)} samples.")
        else:
            _, _, _, _, latencies = results
            
            # Performance calculations
            total_lat_ms = sum(latencies) * 1000
            n_queries = len(latencies)
            # Use mini-batch size if test-mini-batch-size is not specified
            test_bs = args.test_mini_batch_size if args.test_mini_batch_size > 0 else args.mini_batch_size
            total_request_samples = n_queries * test_bs
            num_tables = ln_emb.size
            total_lookups = total_request_samples * num_tables
            
            print("\n" + "="*40)
            print("          PERFORMANCE REPORT            ")
            print("="*40)
            print(f"Total Queries:       {n_queries:,}")
            print(f"Batch Size:          {test_bs}")
            print(f"Total Lookups:       {total_lookups:,}")
            print(f"Total Latency:       {total_lat_ms/1000:.3f} s")
            print(f"Average Latency:     {total_lat_ms / n_queries:.4f} ms")
            print(f"Throughput:          {total_request_samples / (t1 - t0):.2f} samples/s")
            print(f"Perfect Hit Rate:    {perfect_hit / total_lookups:.4%}")
            print(f"Overall Hit Rate:    {total_hit / total_lookups:.4%}")
            print("="*40 + "\n")
    
    print("[DLRM] Execution finished successfully.")


if __name__ == "__main__":
    main()
