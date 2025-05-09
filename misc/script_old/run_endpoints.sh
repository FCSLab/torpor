#!/bin/bash


CLI_NUM=$1
SVR_NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

CUR_SVR=0
for i in `seq 0 $((CLI_NUM-1))`; do
    CLI_ID=$i
    SVR_ID=$CUR_SVR
    # docker run --rm --network=host --ipc=host --mount source=ipc,target=/cuda standalone-client bash /start_with_server_id.sh $CLI_ID $SVR_ID cv_endpoint.py $((CLI_ID+9000)) &
    docker run --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda standalone-client bash /start_with_server_id.sh $CLI_ID $SVR_ID cv_endpoint.py $((CLI_ID+9000)) &

    CUR_SVR=$((CUR_SVR+1))
    CUR_SVR=$((CUR_SVR%SVR_NUM))
done



