**Full logs or/and screenshots can be found at [here](https://drive.google.com/drive/folders/1sr6MMDys4Ta7NfvqWc4ro2xW9fr7pZf4?usp=drive_link) **

# Experimental Procedure

**The order of testing is +group, +Pipeline, +Pinned, from right to left.**

1. launch standalone-server

   ```shell
   # +group
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -it  standalone-server  bash start.sh
     
   # +Pipeline
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -e BUFFER_SIZE=1 \
     -it standalone-server bash start.sh
   # +Pinned  
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -e BUFFER_SIZE=1610612736 \
     -it standalone-server bash start.sh
   ```

2. run test script

   ```shell
   python3 swap_PCIe_latency.py bertqa
   python3 swap_PCIe_latency.py resnet152
   ```

3. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

**For the leftmost baseline**

1. launch standalone-server

   ```shell
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -e BUFFER_SIZE=1610612736 \
     -it standalone-server bash
   ```

2. modify memory_manager.hpp

   ```shell
   cd /gpu-swap/include/server/
   
   nano memory_manager.hpp
   
   control + w
   input: if (src_gpu_id < 0) {
   comment out of the code: if (src_gpu_id < 0) { ... }
   comment out of the code: 
   // wait for the param to be pinned
   // while (model_param_readiness_[signal.second]->at(ptr).first < ParamReadiness_Pinned) {
   //     sync_io_queue_->dequeue();
   //     wait_count++;
   // }
   Then, modify cudaMemcpyAsync() the second line                                 model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].host_addr_,
   control + o, enter and control + x
   
   cd ../../build/
   make
   cd target/
   mv server /server_bin
   cd /server_bin
   ./start.sh
   ```

3. run test script

   ```shell
   python3 swap_PCIe_latency.py bertqa
   python3 swap_PCIe_latency.py resnet152
   ```

4. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```
