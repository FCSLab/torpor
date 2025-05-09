import time
import sys
import os
import zmq
from signal_pb2 import *
import numpy as np

context = zmq.Context(1)
load_socket = context.socket(zmq.PUSH)
load_socket.connect('ipc:///dev/shm/ipc/schedule_load')

def all_to_all(server_num, func_num):
    for i in range(func_num):
        for j in range(server_num):
            load_socket.send_string(f'{i},{j}')
        time.sleep(0.5)

def offline_profile(server_num, func_num):
    func_server_map = {}
    cur_server = 0
    for i in range(func_num):
        func_server_map[i] = cur_server
        cur_server = (cur_server + 1) % server_num
    
    server_load = [0 for _ in range(server_num)]
    func_load = [0 for _ in range(func_num)]
    if os.path.isfile(f'req_arrivals_{func_num}.npy'):
        req_arrivals = np.load(f'req_arrivals_{func_num}.npy')
        for i, f in req_arrivals:
            server_load[func_server_map[f]] += 1
            func_load[int(f)] += 1
        
        print(f'Server load {server_load}')
        func_load_with_idx = [(i, func_server_map[i], f) for i, f in enumerate(func_load)]
        func_load_with_idx.sort(key = lambda x: x[2], reverse = True)
        print(f'Func load {func_load_with_idx[:10]}')

if __name__ == '__main__':
    server_num = 1
    if (len(sys.argv) > 1):
        server_num = int(sys.argv[1])

    client_num = 1
    if (len(sys.argv) > 2):
        client_num = int(sys.argv[2])

    # all_to_all(server_num, client_num)
    offline_profile(server_num, client_num)


