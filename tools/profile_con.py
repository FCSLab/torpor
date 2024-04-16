import time
import os
import sys
import numpy as np
from datetime import datetime

con_start_t = 1673852642.600933
seq_start_t = 1673852690.444143

# con_start_t = 1673851513.3557703
# seq_start_t = 1673851583.5219476

con_start_t = 1674993888.0008426
seq_start_t = 1674994178.0011816

con_d = datetime.fromtimestamp(con_start_t)
con_s = con_d.second + float(con_d.microsecond) / 1000000
seq_d = datetime.fromtimestamp(seq_start_t)
seq_s = seq_d.second + float(seq_d.microsecond) / 1000000
# print(con_s, seq_s)
con_filename = sys.argv[1]
seq_filename = sys.argv[2]

def parse_op_stamp(min_s, start_s, path):
    if not os.path.isfile(path):
        print('Not a file')
        exit(1)
    
    stamps = []
    api_counts = []

    api_recv_info = None
    with open(path, "r") as f:
        for line in f:
            min_num = int(line[15:17])
            stamp = float(line[18:27])
            if min_num == min_s and stamp >= start_s:
                if 'cudnnConvolutionForwardService' in line:
                    # print(stamp, start_s)
                    stamps.append(stamp - start_s)
                
                if 'API received' in line:
                    api_recv_info = (stamp - start_s, line[48:])
                
                if 'API handled' in line:
                    api_counts.append((api_recv_info[0], stamp - start_s, api_recv_info[1]))
    # stamps = [s - stamps[0] for s in stamps]
    return stamps, api_counts

con_stamps, con_api = parse_op_stamp(con_d.minute, con_s, con_filename)
seq_stamps, seq_api = parse_op_stamp(seq_d.minute, seq_s, seq_filename)
print(con_stamps)
print(seq_stamps)

for c, s in zip(con_stamps, seq_stamps):
    print(c, s, c-s)

# print(con_api[-1][1] - con_api[0][0])
# print(seq_api[-1][1] - seq_api[0][0])

# con_sum = sum([e - s for s, e, _ in con_api])
# seq_sum = sum([e - s for s, e, _ in seq_api])
# print(con_sum, seq_sum)

# con_gap = sum([b[0] - a[1] for a, b in zip(con_api[:-1], con_api[1:])])
# seq_gap = sum([b[0] - a[1] for a, b in zip(seq_api[:-1], seq_api[1:])])
# print(con_gap, seq_gap)
