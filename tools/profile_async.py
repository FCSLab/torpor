import time
import os
import sys
import numpy as np
from datetime import datetime

start_t = 1667210193.3627331
end_t = 1667210193.3895915

start_t = 1667214931.9356122
end_t = 1667214931.9632707

start_d = datetime.fromtimestamp(start_t)
start_s = start_d.second + start_d.microsecond / 1000000
end_d = datetime.fromtimestamp(end_t)
end_s = end_d.second + end_d.microsecond / 1000000

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('Not a file')
    exit(1)

query_counts = []

with open(filename, "r") as f:
    for line in f:
        stamp = float(line[18:27])
        if stamp >= start_s and stamp <= end_s:
            if 'size' in line:
                begin_idx = line.find('size') + 5
                count = int(line[begin_idx:])
                query_counts.append(count)

print(f'Total count {sum(query_counts)}, len {len(query_counts)}')
