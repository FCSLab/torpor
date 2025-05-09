import time
import os
import sys
import numpy as np
import pandas as pd
import math

arrival_file = 'req_arrivals_160_trace.npy'
arrivals = np.load(arrival_file)

cur_stamp = 0
func_stamps = {}
for i, f in arrivals:
    cur_stamp += i
    if f not in func_stamps:
        func_stamps[f] = []
    func_stamps[f].append(cur_stamp)

func_keepalive = {}
for f, stamps in func_stamps.items():
    intervals = [ i - j for i, j in zip(stamps[1:], stamps[:-1])]
    func_keepalive[int(f)] = np.percentile(intervals, 99)
#     print(f'func {f}, {np.percentile(intervals, 99)}')

num = arrival_file.split('.')[0].split('_')[2]
with open(f'req_arrivals_{num}_keepalive.npy', 'wb') as f:
    np.save(f, func_keepalive)