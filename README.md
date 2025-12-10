# VineCache
This is an open source repository for our paper

> ** Adaptive and Group-Aware Embedding Caching for Efficient Deep Recommendation Inference**
>

## Brief Introduction

Deep recommendation systems typically leverage embedding caches to accelerate embedding vector (EV) lookups by storing important EVs in fast-access tiers, minimizing inference latency. However, existing caching schemes are inefficient for the embedding cache due to unawareness of its all-or-nothing property, where a single cache miss can severely impact performance. To address this problem, we propose VineCache, an adaptive and group-aware embedding caching scheme that efficiently identifies important EVs across diverse access patterns. VineCache leverages a hit count-based scoring mechanism and a matchmaking rating-inspired weighting design to dynamically capture EV lookup patterns. VineCache further employs multi-level eviction and aging mechanisms to adapt to varying access patterns. Moreover, VineCache employs a reinforcement learning scheme to automatically optimize parameters. Experiments with real-world deep learning recommendation model (DLRM) frameworks show that VineCache outperforms state-of-the-art caching schemes, reducing end-to-end inference latency by 1.14$\times$ on average. 

## Platform
We use a GPU server for evaluation, the detailed information of the server:
* 2 \* 26-core Intel Xeon Gold 6230R CPUs, 1 Tesla V100 GPU
* 6 \* 32GB DDR4 DIMMs, 1 \* 960 GB Samsung SM883 SSD
* Ubuntu 22.04 with Linux kernel version 5.4.0
* Python 3.6.13

## Run DLRM with VineCache

### Clone this repo
```sh
git clone https://anonymous.4open.science/r/VineCache-711C/
cd VineCache
```

### Install RocksDB
https://pyrocksdb.readthedocs.io/en/v0.4/installation.html https://github.com/facebook/rocksdb/blob/master/INSTALL.md
```sh
# cd $home
sudo apt-get install build-essential -y
sudo apt-get install libsnappy-dev zlib1g-dev libbz2-dev libgflags-dev -y
git clone https://github.com/facebook/rocksdb.git
cd rocksdb
make shared_lib  # take 30 mins 
   # make clean
```

### Compile VineCache
```sh
cd ./vinecache-dlrm/cache_algo/VineCache_Cython
python3 setup_VineCache_Cython.py build_ext --inplace
```

### An Example of Running VineCache
```sh
input_data="criteo_kaggle_all"

output_dir="./cache-size=10000/"
mkdir -p $output_dir

numactl --cpunodebind=1 --membind=1 ./bench/dlrm_s_criteo_kaggle.sh \
    "--load-model=model.pth \
    --input-data=./input/$input_data \
    --ev-path=stored_model/$input_data/epoch-00/ev-table \
    --use-gpu=True \
    --inference-only=True \
    --percent-data-for-inference=0.004 \
    --ev-lookup-only=False \
    --use-evstore=True \
    --emb-stor=rocksdb \
    --use-emb-cache=True \
    --cache-algo=vinecache_cython \
    --cache-size=10000 \
    --cache-warmup=True \
    --use-memory-map=True \
    --get-cdf-lat=True \
    --cdf-output-dir=$output_dir \
    --overwrite-db=False" 
```

### Generate Workload For Cache Benchmark
```sh
# Increase --percent-data-for-inference for larger cache workloads

./bench/dlrm_s_criteo_kaggle.sh \
    "--load-model=model.pth \
    --input-data=./input/$input_data \
    --ev-path=stored_model/$input_data/epoch-00/ev-table \
    --inference-only=True \
    --use-memory-map=True \
    --percent-data-for-inference=0.1 \
    --use-evstore=True \
    --overwrite-db=False \
    --trace-inference-workload=True" 
```

## Run Cache Benchmark

### Clone this repo
```sh
git clone https://anonymous.4open.science/r/VineCache-8DBC/
cd VineCache
```

### Install Dependencies
```sh
# Install the default Java Runtime Environment (JRE) and JDK 
cd ./cache-benchmark		
sudo apt update
sudo apt install default-jre
sudo apt install default-jdk

# install maven
sudo apt install maven

mvn help:evaluate -Dexpression=settings.localRepository

# Make sure this folder already exists:
# ~/.m2/repository
cp ./cache-benchmark/repository/* ~/.m2/repository/
```

### An Example of Running Cache Benchmark
```sh
cd ./cache-benchmark
chmod +x benchmark.sh
./benchmark.sh
```
