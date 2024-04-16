import time
import sys
import zmq
from signal_pb2 import *
import subprocess
import requests
import os
import numpy as np

context = zmq.Context(1)
signal_socket = context.socket(zmq.REQ)
signal_addr = 'ipc:///dev/shm/ipc/signal_0'
# signal_addr = 'ipc:///cuda/signal_0'
signal_socket.connect(signal_addr)

def send_signal(req):
    signal_socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket.recv())
    return resp

doLaunch = True
if (len(sys.argv) > 1):
    doLaunch = int(sys.argv[1]) == 0

model_mixed = ['resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201', 'inception', 'efficientnet', 'bertqa']
# model_mixed = ['densenet169', 'densenet201', 'bertqa']
num = 32

if doLaunch:
    for i in range(num):
        req = SignalRequest()
        req.type = ExecuteAfterLoad
        req.function = str(i)

        ack = send_signal(req)
        server_id = ack.resp
        print(f'ExecuteAfterLoad {i} on server {server_id}')
        # p = subprocess.Popen(["bash", "/start_with_server_id.sh", str(i), str(server_id), "cv_endpoint.py", str(9000 + i)])

        model_n = model_mixed[i%len(model_mixed)]
        os.system(f'docker run --rm --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_n} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{i} standalone-client bash /start_with_server_id.sh {i} {server_id} endpoint.py {i + 9000} &')

        if i > 0 and (i + 1) % 4 == 0:
            time.sleep(8)
            for j in range(i - 3, i + 1):
                while True:
                    try:
                        x = requests.get('http://localhost:' + str(9000 + j))
                    except:
                        time.sleep(5)
                    else:
                        break

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

for i in range(num):
    req = SignalRequest()
    req.type = Unload
    req.function = str(i)
    ack = send_signal(req)   

print('Clear up done')

for i in range(num):
    execute_func(i, 0)
    execute_func(i, 0)

# a = np.random.randint(20, size = 50)
# for i in a:
#     execute_func(i, 0)