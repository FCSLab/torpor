#!/bin/bash

# Get input arguments
FUNC_NUM=$1
MINUTES=$2

if [[ -z "$FUNC_NUM" || -z "$MINUTES" ]]; then
    echo "Usage: $0 <FUNC_NUM> <MINUTES>"
    exit 1
fi

ORIG_DIR=$(pwd)

# Run export_keepalive.py
cd ../../tools/
python3 export_keepalive.py req_arrivals_${FUNC_NUM}_${MINUTES}min_trace.npy
sleep 3

# Start baseline_keepalive.py in background
cd "$ORIG_DIR"
echo "Current working directory: $(pwd)"
ls -l baseline_keepalive.py
nohup python3 baseline_keepalive.py -m resnet152 -s 4 -f "$FUNC_NUM" > baseline.log 2>&1 &
sleep 10

# Wait for containers to be ready
echo "Waiting for all standalone-native containers to be ready..."

WAIT_TIMEOUT=10

# Determine how many clients to check
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

echo "All containers are up and responding."
sleep 5

# Run sender_from_trace.py and wait for it to finish
echo "Running sender_from_trace.py..."
python3 sender_from_trace.py "$FUNC_NUM" "$MINUTES" 0

# Check exit status
if [ $? -eq 0 ]; then
    echo "sender_from_trace.py completed successfully."
else
    echo "sender_from_trace.py exited with an error."
    exit 1
fi
sleep 10

# Kill all baseline_keepalive.py processes
echo "Stopping all baseline_keepalive.py processes..."
pkill -f baseline_keepalive.py
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
python3 figure10_analyze_router_log.py ka.log --total_func "$FUNC_NUM"
