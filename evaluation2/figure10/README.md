For Figure 10, when running the 3 test scripts, you need to use the same number of functions and runtime, and each test script will see the results of the log analysis at the end of it.

```bash
chmod +x run_figure10_native.sh
chmod +x run_figure10_ka.sh

# The first parameter represents the number of functions and the second parameter represents the runtime length

# Native
./run_figure10_native.sh 40 5

# Torpor
python3 run_figure10_torpor.py -f 40 -d 5
python3 run_figure10_torpor.py -f 160 -d 5

# INFless-KA
./run_figure10_ka.sh 40 5
./run_figure10_ka.sh 160 5
# If you want to test different runtimes, you need to modify this line at the end of the baseline_keepalive.py
# keep_alive = np.load(f'../../tools/req_arrivals_{func_num}_5min_keepalive.npy', allow_pickle=True).item()
# keep_alive = np.load(f'../../tools/req_arrivals_{func_num}_10min_keepalive.npy', allow_pickle=True).item()
```

