import time
import os
import sys
import numpy as np
from datetime import datetime

start_t = 1665297524.9259744
end_t = 1665297525.4555893 # TCP elasped 0.5296149253845215 vs. 0.0273

start_t = 1665302552.1002724
end_t = 1665302552.5516489 # IPC elasped 0.45137643814086914

# start_t = 1665311680.1759238
# end_t = 1665311680.4175892 # GetDevice-eliminated elasped 0.24166536331176758

# start_t = 1665387857.5170949 
# end_t = 1665387857.6084492 # resnet50 elasped 0.0913543701171875 vs. 0.0108

# start_t = 1665388439.4168828
# end_t = 1665388439.4340706 # vgg16 elasped 0.017187833786010742 vs. 0.0138

# start_t = 1665391546.9069932
# end_t = 1665391547.0033638 # bert elasped 0.09637069702148438 vs. ?

start_d = datetime.fromtimestamp(start_t)
start_s = start_d.second + start_d.microsecond / 1000000
end_d = datetime.fromtimestamp(end_t)
end_s = end_d.second + end_d.microsecond / 1000000

filename = sys.argv[1]
if not os.path.isfile(filename):
    print('Not a file')
    exit(1)

api_calls = []
filters = ['cudaGetDevice', 'cudaGetLastError']

with open(filename, "r") as f:
    for line in f:
        stamp = float(line[18:27])
        if stamp >= start_s and stamp <= end_s:
            if 'Start' in line:
                api_name = line[42:-1]
                start_api_stamp = stamp
            elif 'End' in line:
                api_name = line[40:-1]
                api_calls.append((api_name, stamp - start_api_stamp))
            else:
                print('Warn')
                exit(1)

results = [i[0] for i in api_calls if i[0] not in filters]
print(f'Number of calls {len(results)}')
print(results)
