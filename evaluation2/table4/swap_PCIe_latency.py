# GPU remoting latency
import time
import sys
import zmq
from signal_pb2 import *
import subprocess
import requests
import numpy as np
import os

num = 1
model_n = 'resnet152'
if (len(sys.argv) > 1):
    model_n = sys.argv[1]

launch = True
if (len(sys.argv) > 2):
    launch = int(sys.argv[2]) == 0

context = zmq.Context(1)
signal_socket = context.socket(zmq.REQ)
signal_addr = 'ipc:///dev/shm/ipc/signal_0'
signal_socket.connect(signal_addr)

def send_signal(req):
    signal_socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket.recv())
    return resp

if launch:
    for i in range(num):
        req = SignalRequest()
        req.type = ExecuteAfterLoad
        req.function = str(i)

        ack = send_signal(req)
        server_id = ack.resp
        print(f'ExecuteAfterLoad {i} on server {server_id}')
        os.system(f'docker run --rm --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_n} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{i} standalone-client bash /start_with_server_id.sh {i} {server_id} endpoint.py {i + 9000} &')

    time.sleep(10)

    for i in range(num):
        while True:
            try:
                x = requests.get('http://localhost:' + str(9000 + i))
            except:
                time.sleep(5)
            else:
                break

elasped = {}
end2end_elasped = {}

def execute_func(func, server = -1):
    start_t = time.time()

    req = SignalRequest()
    req.type = Execute
    req.function = str(func)
    if server >= 0:
        req.payload = str(server)

    ack = send_signal(req)
    server_id = ack.resp

    mid_t = time.time()
    x = requests.get('http://localhost:' + str(9000 + func), headers={"cur_server" : str(server_id)})
    end_t = time.time()

    print(f'Execute {func} on {server_id}: {x.text}')
    print(f'[{func}] end2end {end_t - start_t}, signal {mid_t - start_t}, start_t {start_t}')

    start_i = x.text.find('elasped:')
    elasped_str = x.text[start_i + 9:start_i+17]
    elasped[server_id] = elasped[server_id] + [float(elasped_str)] if server_id in elasped else [float(elasped_str)]
    end2end_elasped[server_id] = end2end_elasped[server_id] + [end_t - start_t] if server_id in end2end_elasped else [end_t - start_t]
    time.sleep(1)

src_gpu = 1
execute_func(0, 0)

req = SignalRequest()
req.type = Unload
req.function = str(0)
req.payload = str(0)
ack = send_signal(req)

# execute_func(0, src_gpu)
for i in range(10):
    execute_func(0, 0)

    req = SignalRequest()
    req.type = Unload
    req.function = str(0)
    req.payload = str(0)
    ack = send_signal(req)

elasped_list = elasped[0][1:]
print(f'Latency avg {np.average(elasped_list)}, std {np.std(elasped_list)}')
# print(elasped)
elasped_list = end2end_elasped[0][1:]
print(f'End2End avg {np.average(elasped_list)}, std {np.std(elasped_list)}')

