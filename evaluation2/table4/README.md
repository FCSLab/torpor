For Table 4, all test scripts will execute the 8 models in turn and output a summary of each model's results at the end.

```bash
chmod +x run_figure4_GPU_remoting.sh
chmod +x run_figure4_swap_PCIe.sh
chmod +x run_figure4_swap_NVLink.sh

python3 run_figure4_native.py	# Column 1: Native
./run_figure4_GPU_remoting.sh	# Column 2: GPU remoting
./run_figure4_swap_PCIe.sh	# Column 3: Swap-PCIe
./run_figure4_swap_NVLink.sh	# Column 4: Swap-NVLink
```
