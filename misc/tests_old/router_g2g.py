import time
import sys
import zmq
from signal_pb2 import *
import subprocess
import requests
import numpy as np
from multiprocessing import Pool
import math
import os

context = zmq.Context(1)
signal_socket = context.socket(zmq.REQ)
signal_addr = 'ipc:///dev/shm/ipc/signal_0'
# signal_addr = 'ipc:///cuda/signal_0'
signal_socket.connect(signal_addr)

def send_signal(req, socket):
    socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(socket.recv())
    return resp

selected_server = 0
if (len(sys.argv) > 1):
    selected_server = int(sys.argv[1])

num = 4
if (len(sys.argv) > 2):
    num = int(sys.argv[2])

times = 10
if (len(sys.argv) > 3):
    times = int(sys.argv[3])

launch = True
if (len(sys.argv) > 4):
    launch = len(sys.argv[4]) == 0


if launch:
    for i in range(4):
        req = SignalRequest()
        req.type = ExecuteAfterLoad
        req.function = str(i)

        ack = send_signal(req, signal_socket)
        server_id = ack.resp
        print(f'ExecuteAfterLoad {i} on server {server_id}')
        # p = subprocess.Popen(["bash", "/start_with_server_id.sh", str(i), str(server_id), "cv_endpoint.py", str(9000 + i)])

        # model_n = 'inception' if server_id == 1 else 'resnet152'
        model_n = 'resnet152'
        os.system(f'docker run --rm --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_n} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{i} standalone-client bash /start_with_server_id.sh {i} {server_id} cv_endpoint.py {i + 9000} &')

    time.sleep(10)
    for i in range(4):
        x = requests.get('http://localhost:' + str(9000 + i))
        print(x.text)
        # time.sleep(1)
    
for i in range(4):
    req = SignalRequest()
    req.type = Unload
    req.function = str(i)
    ack = send_signal(req, signal_socket)

elasped = {}

def execute_func(func, server = -1):
    start_t = time.time()

    req = SignalRequest()
    req.type = Execute
    req.function = str(func)
    if server >= 0:
        req.payload = str(server)

    ack = send_signal(req, signal_socket)
    server_id = ack.resp

    mid_t = time.time()
    x = requests.get('http://localhost:' + str(9000 + func), headers={"cur_server" : str(server_id)})
    end_t = time.time()


def query(input):
    func, ser, times = input
    inf_all, e2e_all = [], []
    context_ = zmq.Context(1)
    signal_socket_ = context_.socket(zmq.REQ)
    signal_socket_.connect(signal_addr)

    for _ in range(times):
    # enter_t = time.time()
    # while time.time() - enter_t < times:
        start_t = time.time()
        req = SignalRequest()
        req.type = Execute
        req.function = str(func)
        # req.payload = str(ser)
        ack = send_signal(req, signal_socket_)
        server_id = ack.resp
        # print(f'Execute {func} on server {server_id}')

        mid_t = time.time()
        x = requests.get('http://localhost:' + str(9000 + func), headers={"cur_server" : str(server_id)})
        end_t = time.time()

        print(f'Func {func} on server {server_id} end-to-end time: {end_t - start_t}, issue: {mid_t - start_t}, query: 0, resp: {x.text}')

        if server_id != selected_server:
            req = SignalRequest()
            req.type = Unload
            req.function = str(func)
            req.payload = str(server_id)
            ack = send_signal(req, signal_socket_)

        start_i = x.text.find('elasped:')
        elasped_str = x.text[start_i + 9:start_i+17]
        inf_all.append(float(elasped_str))
        e2e_all.append(end_t - start_t)

        cur_t = time.time()
        sleep_t = round(math.ceil(cur_t) - cur_t, 3)# + 0.01 * func
        time.sleep(sleep_t)

    return ser, inf_all[1:], e2e_all[1:]

test_servers = range(num)

for i in test_servers:
    execute_func(i, selected_server)

with Pool(len(test_servers)) as p:
    all_res = p.map(query, [(s, s, times) for s in test_servers])

inf_all_res, e2e_all_res = [], []
for s, inf, e2e in all_res:
    print(f'[{s}] Latency avg {np.average(inf)}, std {np.std(inf)}')
    print(f'[{s}] E2E latency avg {np.average(e2e)}, std {np.std(e2e)}')

    inf_all_res += inf
    e2e_all_res += e2e

print(f'Latency avg {np.average(inf_all_res)}, std {np.std(inf_all_res)}')
print(f'E2E latency avg {np.average(e2e_all_res)}, std {np.std(e2e_all_res)}')
