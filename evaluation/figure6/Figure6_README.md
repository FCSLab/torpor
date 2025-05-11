**Full logs or/and screenshots can be found at [here](https://drive.google.com/drive/folders/18p5XVL8MehL1gHgo0w3tlIytpPZR3-ZO?usp=sharing)**

**You need to copy the test scripts to run under `torpor/tests`**

# Torpor

## Experimental Procedure

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
   python3 torpor.py resnet152
   
   python3 torpor.py bertqa
   ```

4. After each test, clear up the test

   ```shell
   # clear up test
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

# Torpor w/o batch

## Experimental Procedure

1. launch standalone-server

   ```shell
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -it  standalone-server  bash start.sh
   ```

2. launch standalone-client and enter interactive mode

   ```shell
   docker run -it --rm \
     --cpus=1 \
     -e OMP_NUM_THREADS=1 \
     -e KMP_DUPLICATE_LIB_OK=TRUE \
     --network=host \
     --ipc=host \
     -v /dev/shm/ipc:/cuda \
     --name client-0 \
     standalone-client \
     bash
   ```

   - modify `async_sender.hpp` and remake

     ```shell
     cd /gpu-swap/include/client
     
     nano async_sender.hpp
     control + -
     input: 13
     old: const size_t queryBufferSize = 30;
     modify to: const size_t queryBufferSize = 1;
     control + o, enter, control + x
     
     cd ../../build/
     make
     
     cp src/client/librtclient.so /client_bin
     cd /client_bin
     ldd librtclient.so | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' .
     
     cp /gpu-swap/build/src/client/libmycuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
     cd ..
     ```
   
3. run test script

   ```shell
   # on the test host
   docker cp torpor_without_batch.py client-0:/
   
   model_name=resnet152 python3 torpor_without_batch.py
   model_name=bertqa python3 torpor_without_batch.py
   ```
   
4. After each test, clear up the test

   ```shell
   # clear up test
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

