**Full logs can be found at [here](https://drive.google.com/drive/folders/1Nk2Al3xBwrU844wphwljyeNhk1epImmR?usp=drive_link)**

**You need to first copy the test scripts and sender scripts to run under `torpor/scripts`**

Each data point in Figure 11 needs quite a long time to collect (about one-hour test). We just show an example of running 320 functions on Torpor, and include the parser that analyzes the logs above.

# Experimental Procedure

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
        -it standalone-server bash start.sh
   ```

2. launch 320 functions using 8 various models

   ```shell
   python3 router.py -m mixed -s 4 -f 320 -p sa
   ```

3. send requests from trace

   ```shell
   python3 sender_from_trace.py 320 31 0
   ```

4. clear up test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

# Test Results

After the experimental procedure, you will get a log file.

```bash
python3 figure11_analyze_router_log.py router.log
```

