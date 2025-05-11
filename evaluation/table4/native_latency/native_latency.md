# Readme

Run standalone-native docker container & cp test scripts into the container

```
docker run --gpus all --network=host --ipc=host --name=native standalone-native bash
```

Directly execute each of scripts

```
python3 densenet169_native_execution.py
...
```
