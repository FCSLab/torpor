**Full logs or/and screenshots can be found at [here](https://drive.google.com/drive/folders/1CkOIOt7KQZBpjmyiC0ZtwnGFaSRkLsgQ?usp=drive_link) **

**You need to first copy the test scripts and sender scripts to run under `torpor/scripts`**

# Experimental Procedure

## Native

1. run test script

   ```shell
   # native (no standalone-server)
   # -s 4 is mean to use 4 GPUs
   python3 baseline_native.py -m resnet152 -s 4 -f 40
   ```

2. wait a dozen seconds or so

   ```shell
   # check GPU status
   nvidia-smi
   ```

3. run sender script (send requests from trace)

   ```shell
   python3 sender_from_trace.py 40 5 0 # Generated 2225 requests from trace 
   # rename: router.log -> native_40_5min_from_sapmles_trace.log
   # This sender relies on ../tools/samples.npy
   # After running this script, one trace will be generated. To ensure fairness in testing, the trace is also used in Torpor and INFless-KA.
   cd ../tools
   ls
   req_arrivals_40_5min_trace.npy
   
   python3 sender_from_trace.py 40 10 0 # Generated 3971 requests from trace
   # rename: router.log -> native_40_10min_from_sapmles_trace.log
   # This sender relies on ../tools/samples.npy
   # After running this script, one trace will be generated. To ensure fairness in testing, the trace is also used in Torpor and INFless-KA.
   req_arrivals_40_10min_trace.npy
   ```

4. clear up test

   ```shell
   # Run this command after each test
   docker ps -aq --filter ancestor=standalone-native | xargs docker stop
   ```

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
     -it standalone-server bash start.sh
   ```

2. run test script

   ```shell
   python3 router.py -m resnet152 -s 4 -f 40 -p sa
   
   python3 router.py -m resnet152 -s 4 -f 160 -p sa
   ```

3. run sender script (send requests from trace)

   ```shell
   python3 sender_from_trace.py 40 5 0 # Generated 2225 requests from trace
   # It uses the same trace as native.
   
   python3 sender_from_trace.py 160 10 0 # Generated 15792 requests from trace
   # rename: router.log -> torpor_160_10min_from_sapmles_trace.log
   # This sender relies on ../tools/samples.npy
   # After running this script, one trace will be generated. To ensure fairness in testing, the trace is also used in INFless-KA.
   req_arrivals_160_10min_trace.npy
   ```

4. clear up test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

## INFless-KA

1. run test script

   ```shell
   cd ../tools/
   python3 export_keepalive.py req_arrivals_40_5min_trace.npy
   python3 export_keepalive.py req_arrivals_40_10min_trace.npy
   python3 export_keepalive.py req_arrivals_160_10min_trace.npy
   
   cd ../scripts/
   python3 baseline_keepalive.py -m resnet152 -s 4 -f 40
   python3 baseline_keepalive.py -m resnet152 -s 4 -f 160
   # If you want to test different runtimes, you need to modify this line at the end of the baseline_keepalive file
   # keep_alive = np.load(f'../tools/req_arrivals_{func_num}_5min_keepalive.npy', allow_pickle=True).item()
   # keep_alive = np.load(f'../tools/req_arrivals_{func_num}_10min_keepalive.npy', allow_pickle=True).item()
   ```

2. run sender script (send requests from trace)

   ```shell
   python3 sender_from_trace.py 40 5 0 # Generated 2225 requests from trace
   # It uses the same trace as native and Torpor.
   
   python3 sender_from_trace.py 40 10 0 # Generated 3971 requests from trace
   # It uses the same trace as native.
   
   python3 sender_from_trace.py 160 10 0 # Generated 15792 requests from trace
   # It uses the same trace as Torpor.
   ```

3. clear up test

   ```shell
   docker ps -aq --filter ancestor=standalone-native | xargs docker stop
   ```

# Test Results

After the experimental procedure, you will get a log file.

```shell
python figure10_analyze_router_log.py <logfile name> --total_func 40
python figure10_analyze_router_log.py <logfile name> --total_func 160
```
