#!/bin/bash

# Input arguments
FUNC_NUM=$1
MINUTES=5
# MINUTES=$2

# if [[ -z "$FUNC_NUM" || -z "$MINUTES" ]]; then
#     echo "Usage: $0 <FUNC_NUM> <MINUTES>"
#     exit 1
# fi

if [[ -z "$FUNC_NUM" ]]; then
    echo "Usage: $0 <FUNC_NUM>"
    exit 1
fi

# Start baseline_native.py
nohup python3 baseline_native.py -m resnet152 -s 4 -f "$FUNC_NUM" > baseline.log 2>&1 &
sleep 5

# Wait for containers to be ready
echo "Waiting for all standalone-native containers to be ready..."

WAIT_TIMEOUT=20

if [ "$FUNC_NUM" -gt 72 ]; then
    CHECK_LIMIT=72
else
    CHECK_LIMIT=$FUNC_NUM
fi

for ((i=0; i<$CHECK_LIMIT; i++)); do
    NAME="client-$i"
    PORT=$((9000 + i))
    COUNT=0

    while true; do
        CONTAINER_STATUS=$(docker ps --filter "name=^/$NAME$" --format "{{.Status}}")
        if [[ -z "$CONTAINER_STATUS" ]]; then
            echo "Waiting for container $NAME to appear..."
        elif curl -s "http://localhost:$PORT" > /dev/null; then
            echo "Container $NAME ready on port $PORT."
            break
        else
            echo "Container $NAME found but not responding on port $PORT..."
        fi

        sleep 1
        COUNT=$((COUNT + 1))
        if [ $COUNT -ge $WAIT_TIMEOUT ]; then
            echo "Warning: container $NAME not ready after $WAIT_TIMEOUT seconds. Skipping..."
            break
        fi
    done
done

echo "All containers are up or timeout reached. Proceeding..."
sleep 5

# Run sender_from_trace.py
echo "Running sender_from_trace.py..."
python3 sender_from_trace.py "$FUNC_NUM" "$MINUTES" 0

if [ $? -eq 0 ]; then
    echo "sender_from_trace.py completed successfully."
else
    echo "sender_from_trace.py exited with an error."
    exit 1
fi
sleep 10

# Kill all baseline_native.py processes
echo "Stopping all baseline_native.py processes..."
pkill -f baseline_native.py
sleep 3

# Stop all containers created from standalone-native image
echo "Stopping all standalone-native containers..."
docker ps -aq --filter ancestor=standalone-native | xargs -r docker stop
sleep 3

# Wait until all standalone-native containers are fully stopped
echo "Waiting for all standalone-native containers to stop..."
while docker ps --filter ancestor=standalone-native --format '{{.ID}}' | grep -q .; do
    sleep 1
done
echo "All standalone-native containers have been stopped."
sleep 3

# Run analysis script
echo "Running router log analysis..."
python3 figure10_analyze_router_log.py router.log --total_func "$FUNC_NUM"