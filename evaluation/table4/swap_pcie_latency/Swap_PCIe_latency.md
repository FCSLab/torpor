# 前置操作

1. 启动 standalone-server

   ```shell
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -it  standalone-server  bash start.sh
   ```

2. 编译 protobuf 通信协议文件 signal.proto 为 Python 代码 signal_pb2.py，并移动到/tests中，测试脚本需要导入它并与 standalone-server 通信

   ```shell
   # cd standalone
   cd tests/ 
   bash ../scripts/compile.sh
   ```


3. 运行测试脚本文件（测试脚本中写有自动启动 standalone- client，无需手动启动）

   ```shell
   # 默认模型是 resnet152
   python3 swap_PCIe_latency.py
   
   # 可以通过在后面增加模型名参数的方法，改变测试模型
   # 例如：densenet169
   python3 swap_PCIe_latency.py densenet169
   # densenet169
   # densenet201
   # inception
   # efficientnet
   # resnet50
   # resnet101
   # 默认是 resnet152
   # bertqa
   # 具体模型名看：/tests/endpoint.py
   ```

4. 每次运行完测试脚本后需要关闭 standalone-client 和  standalone-server，下次测试时，需要重启 standalone-server 

   ```shell
   # clear up test
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   
   # 重新启动 standalone-server
   docker run --gpus all --rm --network=host --ipc=host \
     -v /dev/shm/ipc:/cuda \
     -e MEM_LIMIT_IN_GB=25 \
     -e IO_THREAD_NUM=4 \
     -it  standalone-server  bash start.sh
   ```

