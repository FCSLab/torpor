import time
import os
import sys
import numpy as np
from collections import Counter

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('Not a file')
    exit(1)

model_malloc = []
model_memcpy = []

data_malloc = []
data_memcpy = []

model_flag = True

with open(filename, "r") as f:
    for line in f:
        if 'cudaMemcpyAsync' in line:
            kind = int(line[line.find("kind") + 5: line.find("kind") + 6])
            if kind == 1:
                size_start = line.find("count") + 6
                size_end = line.find(' ', size_start)
                size = int(line[size_start:size_end])

                addr_start = line.find("dst") + 4
                addr_end = line.find(' ', addr_start)
                addr = int(line[addr_start:addr_end])
                if model_flag:
                    model_memcpy.append((addr, size))
                else:
                    data_memcpy.append((addr, size))

        elif 'cudaMalloc' in line:
            size_start = line.find("size") + 5
            size_end = line.find(' ', size_start)
            size = int(line[size_start:size_end])

            addr_start = line.find("ptr") + 4
            addr = int(line[addr_start:])

            if model_flag:
                model_malloc.append((addr, size))
            else:
                data_malloc.append((addr, size))
        elif 'cudaDeviceSynchronize' in line:
            model_flag = False
        

model_malloc_count = Counter([i[1] / 1024 /1024 for i in model_malloc])
print(model_malloc_count)

# data_malloc_count = Counter([i[1] / 1024 /1024 for i in data_malloc])
# print(data_malloc_count)

print(sum([i[1] / 1024 /1024 for i in model_malloc]), sum([m_cpy[1] / 1024 / 1024 for m_cpy in model_memcpy]))
print(len(model_malloc), len(model_memcpy))

print(sum([i[1] / 1024 /1024 for i in data_malloc]), sum([m_cpy[1] / 1024 / 1024 for m_cpy in data_memcpy]))
print(len(data_malloc), data_malloc)