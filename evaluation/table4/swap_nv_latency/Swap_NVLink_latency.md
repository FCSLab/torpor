# Readme

**You need to copy the test scripts to run under `torpor/tests`**

1. Make sure NVLink is available for the test host: `nvidia-smi topo -m`

2. launch standalone-server and enter interactive mode

   ```shell
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -it  standalone-server  bash
   ```
   
   - modify `controller.hpp` and remake
   
     ```shell
     cd /gpu-swap/include/server/
     
     nano controller.hpp
     control + -
     input: 426
     # Comment out line 426 and add the following two lines of code
     //  send_load_req(func, gpu_idx);
     if (gpu_idx == 2) { send_load_req(func, gpu_idx, 1); }
     else{ send_load_req(func, gpu_idx); }
     control + o, enter, control + x
     
     cd ../../build/
     make
     cd target/
     mv server /server_bin
     cd /server_bin
     ./start.sh
     ```
   
4. run test script

   ```shell
   python3 swap_NVLink_latency.py densenet169
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
   
5. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

