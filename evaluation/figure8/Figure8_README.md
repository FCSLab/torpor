**Full logs or/and screenshots can be found at [here](https://drive.google.com/drive/folders/1O26NhI_OG-IeMwIQNwIc8YpnTVZZgwwm?usp=sharing) **

**You need to first copy the test scripts and sender scripts to run under `torpor/scripts`**

# Native

## experimental procedure

1. run test script

   ```shell
   # native (no standalone-server)
   python3 baseline_native.py -m resnet152 -s 1 -f 19
   ```

2. wait a dozen seconds or so

   ```shell
   # check GPU status
   nvidia-smi
   ```

3. run sender script

   ```shell
   python3 sender_fix_rate.py 19 5 0 80
   # The fourth parameter is the per-function request rate.
   # You can switch it to 80, 60, 40, 10
   python3 sender_fix_rate.py 19 5 0 60
   python3 sender_fix_rate.py 19 5 0 40
   python3 sender_fix_rate.py 19 5 0 10
   ```

4. clear up test

   ```shell
   # Run this command after each test
   docker ps -aq --filter ancestor=standalone-native | xargs docker stop
   ```

## test results

After the experimental procedure, you will get a log file.

```shell
python3 analyze_router_log.py router.log
```

# Torpor

## experimental procedure

1. launch standalone-server, but need to limit a single GPU

   ```shell
   docker run \
     --gpus device=0 \
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
   # -f 33, 40, 60, 120, 210
   python3 router.py -m resnet152 -s 1 -f 33 -p sa
   # Before running the sender script, you need to wait a bit to make sure that all standalone-clients are up.

   # other tests
   python3 router.py -m resnet152 -s 1 -f 40 -p sa
   python3 router.py -m resnet152 -s 1 -f 60 -p sa
   python3 router.py -m resnet152 -s 1 -f 120 -p sa
   python3 router.py -m resnet152 -s 1 -f 210 -p sa
   ```

4. run sender script

   ```shell
   python3 sender_fix_rate.py 33 5 0 80
   # The first parameter is the number of functions.
   # The second parameter is the runtime length.
   # The fourth parameter is the per-function request rate.
   
   # other tests
   python3 sender_fix_rate.py 40 5 0 60
   python3 sender_fix_rate.py 60 5 0 40
   python3 sender_fix_rate.py 120 5 0 20
   python3 sender_fix_rate.py 210 5 0 10
   ```

5. clear up test

   ```shell
   docker ps -aq --filter ancestor=standalone-client | xargs docker stop
   docker ps -aq --filter ancestor=standalone-server | xargs docker stop
   ```

## test results

After the experimental procedure, you will get a log file.

```shell
python3 figure8_analyze_router_log.py router.log
```
