# Readme

**You need to copy the test scripts to run under `torpor/tests`**

1. launch standalone-server

   ```shell
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -it  standalone-server  bash start.sh
   ```

3. run test script

   ```shell
   python3 GPU_remoting_latency.py densenet169
   # densenet169
   # densenet201
   # inception
   # efficientnet
   # resnet50
   # resnet101
   # resnet152
   # bertqa
   # Model List: /tests/endpoint.py
   ```
   
4. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```
