import time
import os
import sys

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('Not a file')
    exit(1)

outfile = filename.split('/')[-1].split('.')[0] + '_cudnn.txt'

cudaapi = []
with open(filename, "r") as f:
    for line in f:
        if line.find("function ") != -1:
            begin = line.find("function ") + 9
            end = line.find("(", begin)
            if line[begin:end] not in cudaapi:
                cudaapi.append(line[begin:end])

with open(outfile, "a") as wf:
    for i in cudaapi:
        wf.write(i + "\n")
