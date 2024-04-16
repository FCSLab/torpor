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
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', filename='bench.log', filemode='w', level=logging.INFO)

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

times = 10
if (len(sys.argv) > 1):
    times = int(sys.argv[1])

launch = True
if (len(sys.argv) > 2):
    launch = len(sys.argv[2]) == 0

model_mixed = ['densenet169', 'densenet201']
# model_mixed = ['densenet169', 'resnet152']
# model_mixed = ['densenet169', 'bertqa']
# model_mixed = ['resnet152', 'bertqa']

if launch:
    for i in range(len(model_mixed)):
        req = SignalRequest()
        req.type = ExecuteAfterLoad
        req.function = str(i)

        ack = send_signal(req, signal_socket)
        server_id = ack.resp
        print(f'ExecuteAfterLoad {i} on server {server_id}')
        # p = subprocess.Popen(["bash", "/start_with_server_id.sh", str(i), str(server_id), "cv_endpoint.py", str(9000 + i)])

        model_n = model_mixed[i]
        os.system(f'docker run --rm --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_n} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{i} standalone-client bash /start_with_server_id.sh {i} {server_id} endpoint.py {i + 9000} &')

    time.sleep(8)
    for i in range(len(model_mixed)):
        while True:
            try:
                x = requests.get('http://localhost:' + str(9000 + i))
            except:
                time.sleep(5)
            else:
                break
elasped = {}

def query(input):
    func, ser, times = input
    inf_all, e2e_all = [], []
    context_ = zmq.Context(1)
    signal_socket_ = context_.socket(zmq.REQ)
    signal_socket_.connect(signal_addr)

    enter_t = time.time()
    while time.time() - enter_t < times:
        start_t = time.time()
        req = SignalRequest()
        req.type = Execute
        req.function = str(func)
        req.payload = str(ser)
        ack = send_signal(req, signal_socket_)
        server_id = ack.resp
        # print(f'Execute {func} on server {server_id}')

        mid_t = time.time()
        x = requests.get('http://localhost:' + str(9000 + func), headers={"cur_server" : str(server_id)})
        end_t = time.time()

        # print(f'[{func}] Executed on {server_id}: {x.text}')
        logging.info(f'Func {func} on server {ser} end-to-end time: {end_t - start_t}, issue: {mid_t - start_t}, query: 0, resp: {x.text}')

        req = SignalRequest()
        req.type = Unload
        req.function = str(func)
        req.payload = str(ser)
        ack = send_signal(req, signal_socket_)

        start_i = x.text.find('elasped:')
        elasped_str = x.text[start_i + 9:start_i+17]
        inf_all.append(float(elasped_str))
        e2e_all.append(end_t - start_t)

    return ser, inf_all[1:], e2e_all[1:]

test_servers = range(len(model_mixed))
with Pool(len(test_servers)) as p:
    all_res = p.map(query, [(s, s, times) for s in test_servers])

inf_all_res, e2e_all_res = [], []
for s, inf, e2e in all_res:
    logging.info(f'[{s}] Latency avg {np.average(inf)}, std {np.std(inf)}')
    logging.info(f'[{s}] E2E latency avg {np.average(e2e)}, std {np.std(e2e)}')

    inf_all_res += inf
    e2e_all_res += e2e

logging.info(f'Latency avg {np.average(inf_all_res)}, std {np.std(inf_all_res)}')
logging.info(f'E2E latency avg {np.average(e2e_all_res)}, std {np.std(e2e_all_res)}')
