import time
import os
import sys
import zmq
from signal_pb2 import *
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor, wait
import numpy as np

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

client_num = 1
if (len(sys.argv) > 2):
    client_num = int(sys.argv[2])

def launch_client(client_id, server_id):
    req = SignalRequest()
    req.type = ExecuteAfterLoad
    req.function = str(client_id)

    # run all clients on server 0
    ack = send_signal(req, server_id)
    print(f'ExecuteAfterLoad {client_id} signal ack {ack.ack}')

    os.system(f'docker run --rm -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_n} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{client_id} standalone-client bash /start_with_server_id.sh {client_id} {server_id} cv_endpoint.py {client_id + 9000} &')
    # os.system(f'docker run --rm --network=host --ipc=host -e OMP_NUM_THREADS=1 -v /dev/shm/ipc:/cuda standalone-client bash /start_with_server_id.sh {client_id} {server_id} cv_endpoint.py {client_id + 9000} &')
    time.sleep(10)
    
    x = requests.get('http://localhost:' + str(9000 + client_id), headers={"cur_server" : str(server_id)})
    print(f'ExecuteAfterLoad {client_id} on server {server_id} resp: {x.text}')
    time.sleep(1)

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
    time.sleep(1)

    req = SignalRequest()
    req.type = Unload
    req.function = str(func)
    ack = send_signal(req, server_id)
    print(f'Unload {func} on server {server_id} signal ack {ack.ack}')

for i in range(client_num):
    launch_client(i, 0)

elasped = []
for _ in range(2):
    for i in range(client_num):
        start_t = time.time()
        req = SignalRequest()
        req.type = Execute
        req.function = str(i)
        ack = send_signal(req, 0)
        # print(f'Sent execute to {i} signal ack {ack.ack}')

        res = requests.get('http://localhost:' + str(9000 + i)).text
        start_i = res.find('elasped:')
        elasped_str = res[start_i + 9:start_i+17]
        elasped.append(float(elasped_str))

        req = SignalRequest()
        req.type = Unload
        req.function = str(i)
        ack = send_signal(req, 0)
        end_t = time.time()
        print(f'end2end {end_t - start_t} res {res}')

print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')
# np.save(f'{model_n}_{concurrent}.npy', elasped)
