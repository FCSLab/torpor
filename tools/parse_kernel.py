import time
import os
import sys

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('Not a file')
    exit(1)

outfile = filename.split('/')[-1].split('.')[0] + '_kernel.txt'

kernels = []
with open(filename, "r") as f:
    for line in f:
        if line.find("cudaLaunchKernel") != -1:
            begin = line.find("fname") + 6
            end = line.find(" ", begin)
            if line[begin:end] not in kernels:
                kernels.append(line[begin:end])

with open(outfile, "a") as wf:
    for i in kernels:
        wf.write(i + "\n")
