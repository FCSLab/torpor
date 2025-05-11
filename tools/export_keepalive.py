import time
import os
import sys
import numpy as np
import pandas as pd
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('arrival_file', help='Path to arrival trace file (e.g. req_arrivals_40_5min_trace.npy)')
args = parser.parse_args()

arrival_file = args.arrival_file
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
    intervals = [i - j for i, j in zip(stamps[1:], stamps[:-1])]
    func_keepalive[int(f)] = np.percentile(intervals, 99)

# Extract "40_5min" as num
num = '_'.join(arrival_file.split('.')[0].split('_')[2:4])

with open(f'req_arrivals_{num}_keepalive.npy', 'wb') as f:
    np.save(f, func_keepalive)

print(f'Exported keepalive file: req_arrivals_{num}_keepalive.npy')
