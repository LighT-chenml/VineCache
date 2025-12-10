input_data="criteo_kaggle_all"

declare -a arr=("fifo_cython" "lru_cython" "lfu_cython" "evlfu_cython" "vinecache_cython" "vinecache_mfu_cython")

for item in "${arr[@]}"
do
    declare -a arrSize=("37495" "18747" "3749" "1875")

    for size in "${arrSize[@]}"
    do
        output_dir="./cache-size=$size/"
        mkdir -p $output_dir
        
        echo "_________________________________ Working on $item _________________________________"
        numactl --cpunodebind=1 --membind=1 ./dlrm_kaggle.sh \
        "--load-model=model.pth \
        --input-data=./input/$input_data \
        --ev-path=stored_model/$input_data/epoch-00/ev-table \
        --use-gpu=True \
        --inference-only=True \
        --percent-data-for-inference=0.004 \
        --ev-lookup-only=False \
        --use-evstore=True \
        --emb-stor=rocksdb \
        --ev-precs=32 \
        --use-emb-cache=True \
        --cache-algo=$item \
        --cache-size=$size \
        --cache-warmup=True \
        --use-memory-map=True \
        --get-cdf-lat=True \
        --cdf-output-dir=$output_dir \
        --overwrite-db=False" 
    done
done