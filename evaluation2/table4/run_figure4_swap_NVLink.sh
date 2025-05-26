#!/bin/bash

MODELS=("densenet169" "densenet201" "inception" "efficientnet" "resnet50" "resnet101" "resnet152" "bertqa")

REQUIRED_LOGS=(
  "Server 0 -- Set I/O thread num to 4"
  "Server 1 -- Set I/O thread num to 4"
  "Server 2 -- Set I/O thread num to 4"
  "Server 3 -- Set I/O thread num to 4"
)

RESULT_FILE="final_results.txt"
> "$RESULT_FILE"

for MODEL in "${MODELS[@]}"; do
    echo "=============================="
    echo "Testing model: $MODEL"
    echo "=============================="

    CONTAINER_NAME="swap-server-$MODEL"
    LOG_FILE="server_log_$MODEL.txt"

    # Start a fresh container for this model
    docker run --gpus all --network=host --ipc=host \
      -v /dev/shm/ipc:/cuda \
      -e MEM_LIMIT_IN_GB=25 \
      -e IO_THREAD_NUM=4 \
      -dit --name $CONTAINER_NAME standalone-server bash

    # Copy patched controller.hpp into container
    docker cp controller.hpp $CONTAINER_NAME:/gpu-swap/include/server/controller.hpp

    # Recompile inside the container
    docker exec $CONTAINER_NAME bash -c "cd /gpu-swap/build && make"

    # Move server binary and start the service
    docker exec $CONTAINER_NAME bash -c "mv /gpu-swap/build/target/server /server_bin && cd /server_bin && ./start.sh" > "$LOG_FILE" 2>&1 &

    echo "Waiting for server to be ready..."

    for i in {1..60}; do
        MATCHED=0
        for LINE in "${REQUIRED_LOGS[@]}"; do
            if grep -q "$LINE" "$LOG_FILE"; then
                ((MATCHED++))
            fi
        done

        if [ "$MATCHED" -eq 4 ]; then
            echo "Server is ready."
            break
        fi
        sleep 1
    done

    if [ "$MATCHED" -ne 4 ]; then
        echo "Timeout waiting for server readiness logs."
        docker stop $CONTAINER_NAME >/dev/null
        docker rm $CONTAINER_NAME >/dev/null
        exit 1
    fi

    sleep 5

    echo -n "Executing latency test... "
    python3 swap_NVLink_latency.py "$MODEL" > tmp_output_$MODEL.txt 2>&1
    echo "done."
    sleep 3

    LATENCY_LINE=$(grep "Latency avg" tmp_output_$MODEL.txt)
    END2END_LINE=$(grep "End2End avg" tmp_output_$MODEL.txt)

    echo "Model: $MODEL" >> "$RESULT_FILE"
    echo "$LATENCY_LINE" >> "$RESULT_FILE"
    echo "$END2END_LINE" >> "$RESULT_FILE"
    echo >> "$RESULT_FILE"

    # Stop and remove the container
    docker stop $CONTAINER_NAME >/dev/null
    docker rm $CONTAINER_NAME >/dev/null
    docker ps -aq --filter ancestor=standalone-client | xargs docker stop
    sleep 3

    echo
done

echo "=============================="
echo "Swap NVLink: All results collected:"
echo "=============================="
cat "$RESULT_FILE"
