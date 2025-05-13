**Full logs can be found at [here](https://drive.google.com/drive/folders/1Nk2Al3xBwrU844wphwljyeNhk1epImmR?usp=drive_link)**

**You need to first copy the test scripts and sender scripts to run under `torpor/scripts`**

Each data point in Figure 11 needs quite a long time to collect (about one-hour test). Therefore, all commands will be executed in the background.

# Experimental Procedure

## Torpor

1. launch standalone-server (4 GPUs)

   ```shell
   docker run \
       --gpus all \
       --rm \
       --network=host \
       --ipc=host \
       -v /dev/shm/ipc:/cuda \
       -e MEM_LIMIT_IN_GB=25 \
       -e IO_THREAD_NUM=4 \
       -d \
       standalone-server \
       bash start.sh
   # You can check the container logs to determine if the startup is complete or not.
   # docker logs <container_id>
   ```

2. launch 560 functions using 8 various models

   ```shell
   nohup python3 -u router.py -m mixed -s 4 -f 560 -p sa > router_stdout.log 2>&1 &
   # -f 320 400 480 560
   # You can check the number of standalone-cline containers started to determine if the startup is complete or not.
   # For 560, it takes about 30min.
   ```

3. send requests from trace

   ```shell
   nohup python3 sender_from_trace.py 560 31 0 > sender.log 2>&1 &
   # 320 400 480 560
   # This will be executed for 31min per test.
   ```

4. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

## Torpor-FIFO

1. launch standalone-server (4 GPUs)

   ```shell
   docker run \
       --gpus all \
       --rm \
       --network=host \
       --ipc=host \
       -v /dev/shm/ipc:/cuda \
       -e MEM_LIMIT_IN_GB=25 \
       -e IO_THREAD_NUM=4 \
       -d \
       standalone-server \
       bash start.sh
   # You can check the container logs to determine if the startup is complete or not.
   # docker logs <container_id>
   ```

2. launch 560 functions using 8 various models

   ```shell
   nohup python3 -u router.py -m mixed -s 4 -f 560 -p fifo > router_stdout.log 2>&1 &
   # -f 320 400 480 560
   # You can check the number of standalone-cline containers started to determine if the startup is complete or not.
   # For 560, it takes about 30min.
   ```

3. send requests from trace

   ```shell
   nohup python3 sender_from_trace.py 560 31 0 > sender.log 2>&1 &
   # 320 400 480 560
   # This will be executed for 31min per test.
   ```

4. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

## Torpor-Block

1. launch standalone-server (4 GPUs)

   ```shell
   docker run \
       --gpus all \
       --rm \
       --network=host \
       --ipc=host \
       -v /dev/shm/ipc:/cuda \
       -e MEM_LIMIT_IN_GB=25 \
       -e IO_THREAD_NUM=4 \
       -e BLOCK_MANAGER=Fixed \
       -d \
       standalone-server \
       bash start.sh
   # You can check the container logs to determine if the startup is complete or not.
   # docker logs <container_id>
   ```

2. launch 560 functions using 8 various models

   ```shell
   nohup python3 -u router.py -m mixed -s 4 -f 560 -p sa > router_stdout.log 2>&1 &
   # -f 320 400 480 560
   # You can check the number of standalone-cline containers started to determine if the startup is complete or not.
   # For 560, it takes about 30min.
   ```

3. send requests from trace

   ```shell
   nohup python3 sender_from_trace.py 560 31 0 > sender.log 2>&1 &
   # 320 400 480 560
   # This will be executed for 31min per test.
   ```

4. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

## Torpor-LRU

1. launch standalone-server (4 GPUs)

   ```shell
   docker run \
       --gpus all \
       --rm \
       --network=host \
       --ipc=host \
       -v /dev/shm/ipc:/cuda \
       -e MEM_LIMIT_IN_GB=25 \
       -e IO_THREAD_NUM=4 \
       -e EVICT_POLICY=LRU \
       -d \
       standalone-server \
       bash start.sh
   # You can check the container logs to determine if the startup is complete or not.
   # docker logs <container_id>
   ```

2. launch 560 functions using 8 various models

   ```shell
   nohup python3 -u router.py -m mixed -s 4 -f 560 -p sa > router_stdout.log 2>&1 &
   # -f 320 400 480 560
   # You can check the number of standalone-cline containers started to determine if the startup is complete or not.
   # For 560, it takes about 30min.
   ```

3. send requests from trace

   ```shell
   nohup python3 sender_from_trace.py 560 31 0 > sender.log 2>&1 &
   # 320 400 480 560
   # This will be executed for 31min per test.
   ```

4. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

## Torpor-Random

1. launch standalone-server (4 GPUs)

   ```shell
   docker run \
       --gpus all \
       --rm \
       --network=host \
       --ipc=host \
       -v /dev/shm/ipc:/cuda \
       -e MEM_LIMIT_IN_GB=25 \
       -e IO_THREAD_NUM=4 \
       -e SCHEDULE_POLICY=RAND
       -d \
       standalone-server \
       bash start.sh
   # You can check the container logs to determine if the startup is complete or not.
   # docker logs <container_id>
   ```

2. launch 560 functions using 8 various models

   ```shell
   nohup python3 -u router.py -m mixed -s 4 -f 560 -p sa > router_stdout.log 2>&1 &
   # -f 320 400
   # You can check the number of standalone-cline containers started to determine if the startup is complete or not.
   # For 560, it takes about 30min.
   ```

3. send requests from trace

   ```shell
   nohup python3 sender_from_trace.py 560 31 0 > sender.log 2>&1 &
   # 320 400
   # This will be executed for 31min per test.
   ```

4. After each test, clear up the test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

# Test Results

After the experimental procedure, you will get a log file.

```bash
python3 figure11_analyze_router_log.py router.log
```
