**Full logs or/and screenshots can be found at [here](https://drive.google.com/drive/folders/1H_AV6kbP_F6HYYhc5ARmhpqH0E3guNmW?usp=drive_link) **

**You need to first copy the test scripts and sender scripts to run under `torpor/scripts`**

# Experimental Procedure

## Native

1. run test script

   ```shell
   # native (no standalone-server)
   python3 baseline_native.py -m resnet152 -s 4 -f 40
   # -s 4 is mean to use 4 GPUs
   ```

2. wait a dozen seconds or so

   ```shell
   # check GPU status
   nvidia-smi
   ```

3. run sender script

   ```shell
   # To simulate real-world scenarios, we adopted a random approach when sending requests to all functions.
   # The fourth parameter is the per-function request rate. However, if the value is less than 0, a random strategy will be applied.
   # If you want to change the request rate, change "scaled = sampled * 10" in sender_fix_rate.py.
   python3 sender_fix_rate.py 40 5 0 -5 # router13 random*9  Generated 22020
   python3 sender_fix_rate.py 40 5 0 -7 # router15 random*8	Generated 19965
   python3 sender_fix_rate.py 40 5 0 -1 # router10 random*10 Generated 34770
   python3 sender_fix_rate.py 40 5 0 -10 # router17 random*10  Generated 30365
   
   # After running these scripts, four traces will be generated.
   cd ../tools
   ls
   req_arrivals_40_5_-5.npy	req_arrivals_40_5_-7.npy	req_arrivals_40_5_-1.npy	req_arrivals_40_5_-10.npy
   
   # To ensure fairness in testing, the same four traces are also used in Torpor.
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

3. run test script

   ```shell
   # native
   # python3 baseline_native.py -m resnet152 -s 4 -f 40
   # torpor
   python3 router.py -m resnet152 -s 4 -f 40 -p sa
   ```

3. run sender script

   ```shell
   # It is necessary to use the same traces as those used in the native environment.
   # You will see the same number of requests as native.
   python3 sender_fix_rate.py 40 5 0 -5 # router13 random*9  Generated 22020 router13_torpor
   python3 sender_fix_rate.py 40 5 0 -7 # router15 random*8	Generated 19965 router15_torpor
   python3 sender_fix_rate.py 40 5 0 -1 # router10 random*10 Generated 34770 router10_torpor
   python3 sender_fix_rate.py 40 5 0 -10 # router17 random*10  Generated 30365 router17_torpor
   ```

5. clear up test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

# Test Results

After the experimental procedure, you will get a log file.

```shell
python3 figure9_analyze_router_log.py router.log
```
