#!/bin/bash
set -x

# docker run --gpus all -e IO_THREAD_NUM=4 --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda --name cuda_server standalone-server  bash start.sh &

GPU_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
RANGE=$((GPU_NUM-1))

mkdir -p /dev/shm/ipc

for i in `seq 0 $RANGE`; do
    docker run --gpus '"device='${i}'"' --cpus=8 -e SERVER_ID=$i -e IO_THREAD_NUM=6 --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda --name server-${i} standalone-server  bash start.sh &
done

# docker ps -aq | xargs docker stop