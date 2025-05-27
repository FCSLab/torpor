For Figure 10, when running the 3 test scripts, you need to use the same number of functions, and each test script will see the results of the log analysis at the end of it.

```bash
chmod +x run_figure10_native.sh
chmod +x run_figure10_ka.sh

# The parameter represents the number of functions.

# Native
./run_figure10_native.sh 80

# Torpor
python3 run_figure10_torpor.py -f 80

# INFless-KA
./run_figure10_ka.sh 80
```

