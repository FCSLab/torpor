# Readme

Run standalone-native docker container & cp test scripts into the container

```
docker run --gpus all --rm --network=host --ipc=host \
    -v /dev/shm/ipc:/cuda \
    -v /home/torpor/tests:/workspace \
    -e MEM_LIMIT_IN_GB=25 \
    -e IO_THREAD_NUM=4 \
    -it standalone-native \
    python3 /workspace/densenet169_native_execution.py
```
