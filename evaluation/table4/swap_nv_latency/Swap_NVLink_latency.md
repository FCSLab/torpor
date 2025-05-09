# 前置操作

1. 先确定测试主机的 NVLink 可用，运行该命令：`nvidia-smi topo -m`

2. 进入 standalone-server 交互模式

   ```shell
   docker run --gpus all --rm --network=host --ipc=host -v /dev/shm/ipc:/cuda -e MEM_LIMIT_IN_GB=25 -e IO_THREAD_NUM=4 -it  standalone-server  bash
   ```

   - 修改 controller.hpp

     ```shell
     cd /gpu-swap/include/server/
     
     # ls一下，会看到如下文件
     controller.hpp  cuda_server.hpp  kernel_lookup.hpp  memory_manager.hpp  memory_manager_outdated.hpp  model_repo.hpp
     
     nano controller.hpp
     control + -
     输入：426
     注释掉426，并加入下面两行代码
     //  send_load_req(func, gpu_idx);
     if (gpu_idx == 2) { send_load_req(func, gpu_idx, 1); }
     else{ send_load_req(func, gpu_idx); }
     control + O 进行保存 control + x 退出
     
     cd ../../build/
     make
     cd target/
     mv server /server_bin
     cd /server_bin
     ./start.sh		# 启动容器
     ```

3. 编译 protobuf 通信协议文件 signal.proto 为 Python 代码 signal_pb2.py，并移动到/tests中，测试脚本需要导入它并与 standalone-server 通信

   ```shell
   # cd standalone
   cd tests/ 
   bash ../scripts/compile.sh
   ```

4. 运行测试脚本文件（测试脚本中写有自动启动 standalone- client，无需手动启动）

   ```shell
   # 默认模型是 resnet152
   python3 swap_NVLink_latency.py
   
   # 可以通过在后面增加模型名参数的方法，改变测试模型
   # 例如：densenet169
   python3 swap_NVLink_latency.py densenet169
   # densenet169
   # densenet201
   # inception
   # efficientnet
   # resnet50
   # resnet101
   # resnet152
   # bertqa
   # 具体模型名看：/tests/endpoint.py
   ```

5. 每次运行完测试脚本后需要关闭 standalone-client 和  standalone-server，下次测试时，需要重新修改并启动 standalone-server 

   ```shell
   # clear up test
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   
   # 重新执行第二步
   ```


