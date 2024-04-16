#!/bin/bash
set -x

GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
RANGE=$((GPU_NUM-1))
model=$1
func_name=$2

for i in `seq 0 $func_name`; do
    gpu_id=$((i % GPU_NUM))
    docker run --gpus '"device='${gpu_id}'"' --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name=$model --rm  --name client-$i  --network=host --ipc=host standalone-client python endpoint.py $((9000 + $i)) &
    # docker run --gpus '"device='${i}'"' -e model_name=$model -e OMP_NUM_THREADS=1 --rm -p $((9000 + $i)):$((9000 + $i)) standalone-client python /test_endpoint.py $((9000 + $i)) &
done
#     docker run --gpus all --rm -it standalone-server python /gpu-swap/tests/cv_endpoint.cv &

sleep 60

for i in `seq 0 $func_name`; do
    curl localhost:$((9000 + $i))
done