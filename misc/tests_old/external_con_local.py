import time
import os
import sys
from signal_pb2 import *
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np

server_num = 1
if (len(sys.argv) > 1):
    server_num = int(sys.argv[1])

total_client_num = 60

concurrent = server_num
test_servers = [0, 2] if concurrent == 5 else range(concurrent)
pool = ThreadPoolExecutor(max_workers=10)

def query(func):
    start_t = time.time()
    x = requests.get('http://localhost:' + str(9000 + func))
    end_t = time.time()

    res = x.text
    elasped_float = None
    if 'elasped:' in res:
        start_i = res.find('elasped:')
        elasped_float = float(res[start_i + 9:start_i+17])
    print(res)
    print(f'e2e {end_t - start_t}, start_t {start_t}')

    return elasped_float, end_t - start_t

elasped = []
end2end_elasped = []
for _ in range(total_client_num // len(test_servers)):
    fus = [pool.submit(query, i) for i in test_servers]
    for f in fus:
        re = f.result()
        elasped.append(re[0])
        end2end_elasped.append(re[1])
    time.sleep(0.2)

print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')
print(f'E2E latency avg {np.average(end2end_elasped)}, std {np.std(end2end_elasped)}')


# import httpx
# import asyncio
# import time
# import sys

# server_num = 1
# if (len(sys.argv) > 1):
#     server_num = int(sys.argv[1])

# async def get_async(url):
#     async with httpx.AsyncClient() as client:
#         return await client.get(url)

# async def launch():
#     for i in range(10):
#         start_t = time.time()
#         resps = await asyncio.gather(*map(get_async, [f'http://localhost:{9000+i}' for i in range(server_num)]))
#         data = [resp.text for resp in resps]
#         end_t = time.time()
#         print(f'end2end {end_t - start_t}, start_t {start_t}')
#         for d in data:
#             print(d)
        
#         time.sleep(0.1)


# asyncio.run(launch())

from multiprocessing import Pool
import time
import os
import sys
import numpy as np
import requests

server_num = 1
if (len(sys.argv) > 1):
    server_num = int(sys.argv[1])

total_client_num = 60

concurrent = server_num
test_servers = [0, 2] if concurrent == 5 else range(concurrent)

def query(input):
    pro_start_t = time.time()
    func, times = input
    inf_all, e2e_all = [], []
    for _ in range(times):
        start_t = time.time()
        x = requests.get('http://localhost:' + str(9000 + func))
        end_t = time.time()

        res = x.text
        elasped_float = None
        if 'elasped:' in res:
            start_i = res.find('elasped:')
            elasped_float = float(res[start_i + 9:start_i+17])
        
        print(res)
        print(f'e2e {end_t - start_t}, start_t {start_t}')

        inf_all.append(elasped_float), e2e_all.append(end_t - start_t)
        time.sleep(1)
    return inf_all[1:], e2e_all[1:]
    # print(f'Start {pro_start_t}, Latency avg {np.average(e2e_all)}')

with Pool(server_num) as p:
    all_res = p.map(query, [(i, 10) for i in range(server_num)])

inf_all_res, e2e_all_res = [], []
for inf, e2e in all_res:
    inf_all_res += inf
    e2e_all_res += e2e

print(f'Latency avg {np.average(inf_all_res)}, std {np.std(inf_all_res)}')
print(f'E2E latency avg {np.average(e2e_all_res)}, std {np.std(e2e_all_res)}')