**Full logs or/and screenshots can be found at [here](https://drive.google.com/drive/folders/18p5XVL8MehL1gHgo0w3tlIytpPZR3-ZO?usp=sharing)**

**You need to copy the test scripts to run under `torpor/tests`**

# Torpor

## 前置准备

1. 启动 standalone-server

   ```shell
   docker run --gpus all --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda -e MEM_LIMIT_IN_GB=25 -e IO_THREAD_NUM=4 -it  standalone-server  bash start.sh
   ```

2. 编译 protobuf 通信协议文件 signal.proto 为 Python 代码 signal_pb2.py，并移动到/tests中，测试脚本需要导入它并与 standalone-server 通信

   ```shell
   # cd standalone
   cd tests/ 
   bash ../scripts/compile.sh
   ```

3. 在测试主机上，运行测试脚本

   ```shell
   python3 torpor.py resnet152
   # 每次运行测试脚本前，都需要关掉standalone-client 和 standalone-server 并重复前置准备
   python3 torpor.py bertqa
   ```

4. 测试完毕后记得关掉 standalone-client 和 standalone-server

   ```shell
   # clear up test
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

# Torpor w/o batch

## 前置准备

1. 启动 standalone-server

   ```shell
   docker run --gpus all --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda -e MEM_LIMIT_IN_GB=25 -e IO_THREAD_NUM=4 -it  standalone-server  bash start.sh
   ```

2. 手动启动 standalone-client，并进入交互模式

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

   - 修改 async_sender.hpp，重新 make 并将新 make 产生的文件进行替换

     ```shell
     cd /gpu-swap/include/client
     
     # 输入ls，显示如下内容
     async_sender.hpp  cuda_client.hpp
     
     nano async_sender.hpp
     control + -
     输入：13
     原来是：const size_t queryBufferSize = 30;
     修改为：const size_t queryBufferSize = 1;
     control + O 进行保存 control + x 退出
     
     cd ../../build/
     make
     
     # 根据 dockerfiles/client.Dockerfile 里面的逻辑替换新make的两个文件
     cp src/client/librtclient.so /client_bin
     cd /client_bin
     ldd librtclient.so | grep "=> /" | awk '{print $3}' | xargs -I '{}' cp -v '{}' .
     
     cp /gpu-swap/build/src/client/libmycuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
     cd ..
     ```

3. 在 standalone-client 容器中运行测试脚本

   ```shell
   # 将测试脚本 和 endpoint.py 从测试主机移动至 standalone-client 容器中
   docker cp endpoint.py client-0:/
   docker cp torpor_without_batch.py client-0:/
   
   # 运行测试脚本
   model_name=resnet152 python3 test.py
   # 每次运行测试脚本前，都需要关掉standalone-client 和 standalone-server 并重复前置准备
   model_name=bertqa python3 test.py
   ```

4. 测试完毕后记得关掉 standalone-client 和 standalone-server

   ```shell
   # clear up test
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

