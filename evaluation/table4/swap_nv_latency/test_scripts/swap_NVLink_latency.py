'''
Assumption: there is nvlink between gpu_1 and gpu_2

First load model from host to gpu_1, then start formal test: load from gpu1 to gpu2 in each request

Note: before doing this experiment, change the code in controller.hpp (standalone-server: /gpu-swap/include/server/controller.hpp）

    before change:

        else if (req.type() == RequestType::Execute){
            .....
            if (hint.size() > 0) {                
                ......
                send_load_req(func, gpu_idx);
                ......
                }
        }
    
    after change:

        else if (req.type() == RequestType::Execute){
            .....
            if (hint.size() > 0) {                
                ......
                if (gpu_idx == 2) { send_load_req(func, gpu_idx, 1); }  // 强制从 GPU 1 copy 到 GPU 2   
                else{ send_load_req(func, gpu_idx); }   
                ......
                }
        }
'''
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

# from host to gpu1
execute_func(0, 1)

for i in range(10):
    # from gpu1 to gpu2 (transfer by nvlink)
    execute_func(0, 2)

    # unload after each request, so next time will reload from gpu1
    req = SignalRequest()
    req.type = Unload
    req.function = str(0)
    req.payload = str(0)
    ack = send_signal(req)

elasped_list = elasped[2][1:]
print(f'Latency avg {np.average(elasped_list)}, std {np.std(elasped_list)}')

elasped_list = end2end_elasped[2][1:]
print(f'End2End avg {np.average(elasped_list)}, std {np.std(elasped_list)}')


