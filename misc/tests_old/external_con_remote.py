import time
import os
import sys
import zmq
from signal_pb2 import *
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np
from multiprocessing import Pool

do_swap = True

class SocketCache():
    def __init__(self, context):
        self.context = context
        self.socket_cache = {}

    def get(self, addr):
        if addr in self.socket_cache:
            return self.socket_cache[addr]
        else:
            socket = self.context.socket(zmq.REQ)
            socket.connect(addr)
            self.socket_cache[addr] = socket
            return socket


prefix = 'ipc:///dev/shm/ipc'
context = zmq.Context(1)
socket_cache = SocketCache(context)

def send_signal(req, server_id):
    signal_socket = socket_cache.get(f'{prefix}/signal_{server_id}')
    signal_socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket.recv())
    return resp

update_socket = context.socket(zmq.PULL)
update_socket.bind(f'{prefix}/scheduler')

model_n = 'resnet152'
if (len(sys.argv) > 1):
    model_n = sys.argv[1]

server_num = 1
if (len(sys.argv) > 2):
    server_num = int(sys.argv[2])

skip_launch = False
concurrent = server_num
if (len(sys.argv) > 3):
    skip_launch = True
    concurrent = int(sys.argv[3])


# client_num = 1
# if (len(sys.argv) > 2):
#     client_num = int(sys.argv[2])

def launch_client(client_id, server_id, swap=do_swap):
    req = SignalRequest()
    req.type = ExecuteAfterLoad
    req.function = str(client_id)

    # run all clients on server 0
    ack = send_signal(req, server_id)
    print(f'ExecuteAfterLoad {client_id} signal ack {ack.ack}')

    os.system(f'docker run --rm --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_n} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{client_id} standalone-client bash /start_with_server_id.sh {client_id} {server_id} cv_endpoint.py {client_id + 9000} &')
    # os.system(f'docker run --rm --network=host --ipc=host -e OMP_NUM_THREADS=1 -v /dev/shm/ipc:/cuda standalone-client bash /start_with_server_id.sh {client_id} {server_id} cv_endpoint.py {client_id + 9000} &')
    time.sleep(8)
    
    x = requests.get('http://localhost:' + str(9000 + client_id), headers={"cur_server" : str(server_id)})
    print(f'ExecuteAfterLoad {client_id} on server {server_id} resp: {x.text}')

    # time.sleep(1)
    if swap:
        req = SignalRequest()
        req.type = Unload
        req.function = str(client_id)
        ack = send_signal(req, server_id)
        print(f'Unload {client_id} on server {server_id} signal ack {ack.ack}')

def execute(func, server_id):
    req = SignalRequest()
    req.type = Execute
    req.function = str(func)
    ack = send_signal(req, server_id)
    print(f'Execute {func} on server {server_id} signal ack {ack.ack}')

    x = requests.get('http://localhost:' + str(9000 + func), headers={"cur_server" : str(server_id)})
    print(f'Execute {func} on server {server_id} resp: {x.text}')

    # time.sleep(1)
    req = SignalRequest()
    req.type = Unload
    req.function = str(func)
    ack = send_signal(req, server_id)
    print(f'Unload {func} on server {server_id} signal ack {ack.ack}')

poller = zmq.Poller()
poller.register(update_socket, zmq.POLLIN)

if not skip_launch:
    for i in range(server_num):
        launch_client(i, i)

test_servers = [0, 2] if concurrent == 5 else range(concurrent)
total_client_num = 40

# pool = ThreadPoolExecutor(max_workers=len(test_servers))
# elasped = []
# end2end_elasped = []
# for _ in range(total_client_num // len(test_servers)):
#     def query(func, swap=do_swap):
#         start_t = time.time()
#         req = SignalRequest()
#         req.type = Execute  
#         req.function = str(func)
#         ack = send_signal(req, func)
    
#         mid_t = time.time()
#         x = requests.get('http://localhost:' + str(9000 + func))
#         end_t = time.time()

#         res = x.text
#         start_i = res.find('elasped:')
#         elasped_str = res[start_i + 9:start_i+17]
#         print(f'[{func}] {res}')
#         print(f'[{func}] end2end {end_t - start_t}, signal {mid_t - start_t}, start_t {start_t}')

#         if swap:
#             req = SignalRequest()
#             req.type = Unload
#             req.function = str(func)
#             ack = send_signal(req, func)

#         return float(elasped_str), end_t - start_t 

#     fus = [pool.submit(query, i) for i in test_servers]
#     for f in fus:
#         re = f.result()
#         elasped.append(re[0])
#         end2end_elasped.append(re[1])
#     time.sleep(1)

# print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')
# print(f'E2E latency avg {np.average(end2end_elasped)}, std {np.std(end2end_elasped)}')
# # np.save(f'{model_n}_{concurrent}.npy', elasped)

def query(input):
    func, times = input
    inf_all, e2e_all = [], []
    context_ = zmq.Context(1)
    socket_cache_ = SocketCache(context_)

    def send_signal_internel(req):
        signal_socket = socket_cache_.get(f'{prefix}/signal_{func}')
        signal_socket.send(req.SerializeToString())
        resp = SignalAck()
        resp.ParseFromString(signal_socket.recv())
        return resp

    for _ in range(times):
        start_t = time.time()
        req = SignalRequest()
        req.type = Execute  
        req.function = str(func)
        ack = send_signal_internel(req)

        mid_t = time.time()
        x = requests.get('http://localhost:' + str(9000 + func))
        end_t = time.time()

        res = x.text
        start_i = res.find('elasped:')
        elasped_str = res[start_i + 9:start_i+17]
        print(f'[{func}] {res}')
        print(f'[{func}] end2end {end_t - start_t}, signal {mid_t - start_t}, start_t {start_t}')
        inf_all.append(float(elasped_str)) 
        e2e_all.append(end_t - start_t)

        if func%2 == 0:
            req = SignalRequest()
            req.type = Unload
            req.function = str(func)
            ack = send_signal_internel(req)
        
        time.sleep(1)
    # print(f'Latency avg {np.average(inf_all)}, std {np.std(inf_all)}')
    # print(f'E2E latency avg {np.average(e2e_all)}, std {np.std(e2e_all)}')
    return inf_all[1:], e2e_all[1:]

with Pool(len(test_servers)) as p:
    all_res = p.map(query, [(i, total_client_num // len(test_servers)) for i in test_servers])

inf_all_res, e2e_all_res = [], []
for inf, e2e in all_res:
    inf_all_res += inf
    e2e_all_res += e2e

print(f'Latency avg {np.average(inf_all_res)}, std {np.std(inf_all_res)}')
print(f'E2E latency avg {np.average(e2e_all_res)}, std {np.std(e2e_all_res)}')