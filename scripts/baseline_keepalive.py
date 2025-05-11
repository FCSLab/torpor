import time
import sys
import os
import zmq
from signal_pb2 import *
import numpy as np
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor
import argparse
import queue
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue as pqueue
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', filename='ka.log', filemode='w', level=logging.INFO)

class SocketCache():
    def __init__(self, context):
        self.context = context
        self.socket_cache = {}

    def get(self, addr, typ=zmq.REQ):
        if addr in self.socket_cache:
            return self.socket_cache[addr]
        else:
            socket = self.context.socket(typ)
            socket.connect(addr)
            self.socket_cache[addr] = socket
            return socket

model_mixed = ['resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201', 'inception', 'efficientnet', 'bertqa']
def get_mixed_model_name(client_id):
    model_idx = client_id % len(model_mixed)
    return model_mixed[model_idx]

def get_mixed_model_level(client_id):
    model_idx = client_id % len(model_mixed)
    if model_idx < 3:
        return 1
    elif model_idx < 7 and model_idx >= 3:
        return 0
    elif model_idx == 7:
        return 2
    return 1

def get_manager_ipc(manager_id):
    return 'ipc:///dev/shm/ipc/manager_' + str(manager_id)

def container_manager(server_id):
    global bert_function
    logging.info(f'Start manager {server_id}')

    context = zmq.Context(1)
    manager_puller = context.socket(zmq.PULL)
    manager_puller.bind(get_manager_ipc(server_id))

    router_pusher = context.socket(zmq.PUSH)
    router_pusher.connect(f'ipc:///dev/shm/ipc/router')

    poller = zmq.Poller()
    poller.register(manager_puller, zmq.POLLIN)

    while True:
        socks = dict(poller.poll(timeout=1))

        if manager_puller in socks and socks[manager_puller] == zmq.POLLIN:
            obj = manager_puller.recv_pyobj()
            flag, msg = obj
            if flag == '1': # execute
                func_id, issue_t, arrival_t = msg
                
                # logging.info(f'Handle request {func_id}')
                start_t = time.time()
                try:
                    x = requests.get('http://localhost:' + str(9000 + func_id), headers={"cur_server" : str(server_id), "batch_size" : str(len(arrival_t))}, timeout=5)
                except Exception as e:
                    logging.info(f'Func {func_id} on server {server_id} timeout')
                    router_pusher.send_pyobj(('2', func_id))
                else:
                    end_t = time.time()
                    for arr in arrival_t:
                        logging.info(f'Func {func_id} batch size {len(arrival_t)} on server {server_id} end-to-end time: {end_t - arr}, issue: {end_t - issue_t}, query: {end_t - start_t}, resp: {x.text}')
                    router_pusher.send_pyobj(('1', (func_id, [end_t - arr for arr in arrival_t])))

class Router():
    def __init__(self, server_num, func_num):
        self.context = zmq.Context(1)

        self.service_socket = self.context.socket(zmq.PULL)
        self.service_socket.bind('ipc:///dev/shm/ipc/externel_service')

        self.router_socket = self.context.socket(zmq.PULL)
        self.router_socket.bind('ipc:///dev/shm/ipc/router')

        self.func_num = func_num
        self.func_stat = {i: [0, 0, 0] for i in range(func_num)}

        self.func_avail = {}
        self.func_server_map = {}
        self.server_footprint = {i: 0 for i in range(server_num)}

        self.server_num = server_num
        self.schedule_queue = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.sockets = SocketCache(self.context)

        launch_batch = [ (i % server_num, i) for i in range(func_num) if i < server_num * max_model_num_per_server]            
        self.launch_models(launch_batch)
        logging.info(f'Init launching {launch_batch}')

        self.timeout_func = set()
        for i in range(server_num):
            reader_p = Process(target=container_manager, args=(i,))
            reader_p.start()
        logging.info('Start schedule')

    def get_server_id(self, client_id):
        if client_id in self.func_server_map:
            return self.func_server_map[client_id][0]
        else: 
            return None

    def launch_models(self, server_model_ids):

        def inner_launch(self, server_model_ids):
            for server_id, model_id in server_model_ids:
                self.server_footprint[server_id] += 1
                # os.system(f'docker run --gpus \'"device={server_id}"\' --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_name} --rm  --name client-{model_id}  --network=host --ipc=host standalone-client python endpoint.py {9000 + model_id} &')
                os.system(f'docker run --gpus \'"device={server_id}"\' --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_name} --rm  --name client-{model_id}  --network=host --ipc=host standalone-native python endpoint.py {9000 + model_id} &')
                if model_id not in self.func_stat:
                    self.func_stat[model_id] = [0, 0, 0]
            
            # poll the endpoint until it is ready
            for server_id, model_id in server_model_ids:
                while True:
                    try:
                        x = requests.get('http://localhost:' + str(9000 + model_id), timeout=1)
                    except Exception as e:
                        pass
                    else:
                        break
                self.func_server_map[model_id] = [server_id, time.time()]
                self.func_avail[model_id] = True
        self.pool.submit(inner_launch, self, server_model_ids)

    def terminate_models(self, model_ids):
        def inner_terminate(self, model_ids):
            client_ids = ' '.join(['outdated-' + str(model_id) for model_id in model_ids])
            server_record = {i: 0 for i in range(server_num)}

            for model_id in model_ids:
                server_record[self.func_server_map[model_id][0]] += 1
                del self.func_server_map[model_id]
                self.func_avail[model_id] = False
                os.system(f'docker rename client-{model_id} outdated-{model_id}')
            os.system(f'docker stop -t 1 {client_ids}')

            for server_id, num in server_record.items():
                self.server_footprint[server_id] -= num
        self.pool.submit(inner_terminate, self, model_ids)

    def baseline_schedule(self):
        req_queue = []
        def has_req():
            return len(req_queue) > 0
        
        def insert_req(f, arr):
            req_queue.append((f, arr))
        
        def get_req(out=True):
            if out:
                return req_queue.pop(0)
            else:
                return req_queue[0] if len(req_queue) > 0 else None

        def batch_check(f):
            req = get_req(False)
            if req is None or req[0] != f:
                return False
            return True

        wait_func_req = {}
        wait_func = []
        pend_func = []
        def add_func_req(f, arr):
            if f not in wait_func_req:
                wait_func_req[f] = []
                wait_func.append(f)
            wait_func_req[f].append(arr)


        logging.info('Listen on schedule queue using FIFO policy')

        while True:                
            try:
                sched_typ, sched_msg = self.schedule_queue.get(timeout=0.001)
            except queue.Empty:
                if has_req():
                    req = get_req()
                    if req is None:
                        continue
                    
                    f, arr = req
                    # batching
                    arrs = [arr]
                    while batch_check(f) and len(arrs) < batch_size_limit:
                        f, arr = get_req()
                        arrs.append(arr)

                    issue_t = time.time()
                    self.func_avail[f] = False
                    self.func_server_map[f][1] = issue_t
                    self.sockets.get(get_manager_ipc(self.get_server_id(f)), zmq.PUSH).send_pyobj(('1', (f, issue_t, arrs)))

            else:
                if sched_typ == 0: # requests
                    target_func, arr_t = sched_msg
                    if target_func in self.func_server_map:
                        insert_req(target_func, arr_t)
                        logging.info(f'Inserted request {target_func}, cur_len: {len(req_queue)}')
                    else:
                        add_func_req(target_func, arr_t)
                        logging.info(f'Added terminated function request {target_func}')

                elif sched_typ == 1: # update queue
                    count = len([f for f, stat in self.func_stat.items() if stat[1] > 0])
                    if count <= 0:
                        continue

                    slo_ratio = len([f for f, stat in self.func_stat.items() if stat[1] > 0 and stat[0] / stat[1] >= slo_percentile]) / len(self.func_stat)
                    logging.info(f'FIFO update. slo_ratio: {slo_ratio}, total_count: {len(self.func_stat)}, func_count: {count}')

                    # check keep-alive
                    func_to_terminate = []
                    for f, (server_id, last_t) in self.func_server_map.items():
                        if time.time() - last_t > keep_alive[f]:
                            func_to_terminate.append(f)
                    if len(func_to_terminate) > 0:
                        self.terminate_models(func_to_terminate)
                        logging.info(f'Terminating functions {func_to_terminate}')
                    
                    # check if we need to launch new models
                    avail_server_res = [(s,  max_model_num_per_server - fp) for s, fp in self.server_footprint.items() if fp < max_model_num_per_server]
                    if len(avail_server_res) > 0 and len(wait_func) > 0:
                        func_to_launch = []
                        for s, quota in avail_server_res:
                            while len(wait_func) > 0 and quota > 0:
                                f = wait_func.pop(0)
                                quota -= 1
                                func_to_launch.append((s, f))
                        self.launch_models(func_to_launch)
                        for _, f in func_to_launch:
                            pend_func.append(f)
                        logging.info(f'Launching functions {func_to_launch}')

                    # check pending functions
                    for f in pend_func[:]:
                        if f in self.func_server_map:
                            for req in wait_func_req[f]:
                                insert_req(f, req)
                            del wait_func_req[f]
                            pend_func.remove(f)
                            logging.info(f'Pending function {f} is ready')


    
    def run(self):
        self.pool.submit(self.poll_exterel)
        self.baseline_schedule()

    def poll_exterel(self):

        poller = zmq.Poller()
        poller.register(self.service_socket, zmq.POLLIN)
        poller.register(self.router_socket, zmq.POLLIN)
        
        start_t = time.time()
        end_t = time.time()
        while True:
            socks = dict(poller.poll(timeout=1))
            
            if self.service_socket in socks and socks[self.service_socket] == zmq.POLLIN:
                target_func = int(self.service_socket.recv())
                if target_func >= 0:
                    cur_t = time.time()
                    # logging.info(f'Get request {target_func} at {cur_t}')
                    if target_func in self.timeout_func:
                        logging.info(f'Ignore unhealthy func {target_func}')
                    else:
                        self.schedule_queue.put((0, (target_func, cur_t)))
                else:
                    logging.info(f'Unknown target func {target_func}')


            if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
                obj = self.router_socket.recv_pyobj()
                flag, sched_msg = obj

                if flag == '1':
                    func_id, lats = sched_msg
                    for lat in lats:
                        self.func_stat[func_id][2] += lat
                        self.func_stat[func_id][1] += 1
                        self.func_stat[func_id][0] = self.func_stat[func_id][0] + 1 if lat <= latency_ddl else self.func_stat[func_id][0]
                    self.func_avail[func_id] = True
                
                elif flag == '2':
                    func_id = sched_msg
                    self.timeout_func.add(func_id)
            
            end_t = time.time()
            if end_t - start_t > 2: # adjust request queue
                start_t = end_t
                self.schedule_queue.put((1, None))

parser = argparse.ArgumentParser()
parser.add_argument('-m', type=str, default='resnet152')
parser.add_argument('-s', type=int, default=1)
parser.add_argument('-f', type=int, default=1)
parser.add_argument('-t', type=int, default=98)
parser.add_argument('-d', type=int, default=100)
parser.add_argument('-b', type=int, default=1)

args = parser.parse_args()

model_name = args.m
server_num = args.s
func_num = args.f
slo_percentile = args.t / 100
latency_ddl = args.d / 1000
batch_size_limit = args.b

max_model_num_per_server = 18 # hardcoded for resnet152 in a GPU with 32 GB memory
keep_alive = np.load(f'../tools/req_arrivals_{func_num}_keepalive.npy', allow_pickle=True).item()

router = Router(server_num, func_num)
router.run()

