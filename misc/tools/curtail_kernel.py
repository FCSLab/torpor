import time
import os
import sys

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('Not a file')
    exit(1)

outfile = filename.split('/')[-1].split('.')[0] + '_curtail.txt'

kernels = {}
with open(filename, "r") as f:
    for line in f:
        items = line.strip().split(',')
        kernels[items[0]] = items[1:]

with open(outfile, "a") as wf:
    for k, args in kernels.items():
        wf.write(k)
        for arg in args:
         wf.write("," + arg)
        wf.write("\n")