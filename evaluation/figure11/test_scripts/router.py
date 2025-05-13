import time
import sys
import os
import zmq
from signal_pb2 import *
import subprocess
import requests
from concurrent.futures import ThreadPoolExecutor
import argparse
import queue
import multiprocessing
from multiprocessing import Process
from multiprocessing import Queue as pqueue
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', filename='router.log', filemode='w', level=logging.INFO)

bert_ddl = 0.2
bert_function = set()
bert_batch_size = 1

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

def get_manager_ipc(manager_id):
    return 'ipc:///dev/shm/ipc/manager_' + str(manager_id)

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

            if flag == '0':
                client_id = msg
                while True:
                    logging.info(f'Try launching {client_id} on server {server_id}')
                    is_success = False
                    if model_name == 'mixed':
                        mod_n = get_mixed_model_name(client_id)
                        if mod_n == 'bertqa':
                            bert_function.add(client_id)
                        os.system(f'docker run --rm --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={mod_n} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{client_id} standalone-client bash /start_with_server_id.sh {client_id} {server_id} endpoint.py {client_id + 9000} &')
                    else:
                        os.system(f'docker run --rm --cpus=1 -e OMP_NUM_THREADS=1 -e KMP_DUPLICATE_LIB_OK=TRUE -e model_name={model_name} --network=host --ipc=host -v /dev/shm/ipc:/cuda --name client-{client_id} standalone-client bash /start_with_server_id.sh {client_id} {server_id} endpoint.py {client_id + 9000} &')
                    time.sleep(8)
                    for attempt in range(4):
                        try:
                            x = requests.get('http://localhost:' + str(9000 + client_id), headers={"cur_server" : str(server_id), "batch_size": str(bert_batch_size if client_id in bert_function else batch_size_limit)}, timeout=5)
                            # x = requests.get('http://localhost:' + str(9000 + client_id), headers={"cur_server" : str(server_id), "batch_size": str(1)}, timeout=5)
                            logging.info(f'ExecuteAfterLoad {client_id} on server {server_id} resp: {x.text}')
                            router_pusher.send_pyobj(('0', client_id))
                        except:
                            logging.info(f'ExecuteAfterLoad {client_id} on server {server_id} timeout at attempt {attempt}')
                            time.sleep(5)
                        else:
                            is_success = True
                            break
                    if is_success:
                        break
        
            elif flag == '1': # execute
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
    def __init__(self, server_num, func_num=None, sa_policy=True):
        self.context = zmq.Context(1)

        self.service_socket = self.context.socket(zmq.PULL)
        self.service_socket.bind('ipc:///dev/shm/ipc/externel_service')

        self.router_socket = self.context.socket(zmq.PULL)
        self.router_socket.bind('ipc:///dev/shm/ipc/router')

        self.route_policy = self.sa_schedule if sa_policy else self.baseline_schedule

        self.func_num = 0
        self.func_stat = {}
        self.func_avail = {}
        self.high_func, self.low_func = set(), set()
        if func_num is not None and func_num > 0:
            self.func_num = func_num
            self.func_stat = {i: [0, 0, 0] for i in range(func_num)}
            self.func_avail = {i: True for i in range(func_num)}
            self.high_func = set(range(func_num))
            if model_name == 'mixed':
                for i in range(func_num):
                    if i % 8 == 7:
                        bert_function.add(i)

        self.server_num = server_num
        self.schedule_queue = queue.Queue()
        self.pool = ThreadPoolExecutor(max_workers=2)

        self.timeout_func = set()
        self.sockets = SocketCache(self.context)
        for i in range(server_num):
            reader_p = Process(target=container_manager, args=(i,))
            reader_p.start()
        logging.info('Start schedule')

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
                    batch_size_upper = bert_batch_size if f in bert_function else batch_size_limit
                    while batch_check(f) and len(arrs) < batch_size_upper:
                        f, arr = get_req()
                        arrs.append(arr)
                    req = SignalRequest()
                    req.type = Execute
                    req.function = str(f)
                    server_id = self.send_signal(req, self.sockets)

                    issue_t = time.time()
                    self.func_avail[f] = False
                    self.sockets.get(get_manager_ipc(server_id), zmq.PUSH).send_pyobj(('1', (f, issue_t, arrs)))
                    logging.info(f'Scheduled request {f} to server {server_id}')

            else:
                if sched_typ == 0: # requests
                    target_func, arr_t = sched_msg
                    insert_req(target_func, arr_t)
                    logging.info(f'Inserted request {target_func}, cur_len: {len(req_queue)}')
                elif sched_typ == 1: # update queue
                    count = len([f for f, stat in self.func_stat.items() if stat[1] > 0])
                    if count <= 0:
                        continue

                    slo_ratio = len([f for f, stat in self.func_stat.items() if stat[1] > 0 and stat[0] / stat[1] >= slo_percentile]) / count
                    logging.info(f'FIFO update. slo_ratio: {slo_ratio}, queued_req: {len(req_queue)}')

    
    def sa_schedule(self):

        high_req_queue, low_req_queue = [], []
        def has_req():
            return len(high_req_queue) > 0 or len(low_req_queue) > 0

        def calc_metric(f):
            return (self.func_stat[f][2] / self.func_stat[f][1]) * (slo_percentile * self.func_stat[f][1] - self.func_stat[f][0]) if self.func_stat[f][1] > 0 else 0

        def insert_req(f, arr):
            if f in self.high_func:
                start_f, end_f = 0, len(high_req_queue)
                while start_f < end_f:
                    mid_f = (start_f + end_f) // 2
                    if calc_metric(high_req_queue[mid_f][0]) < calc_metric(f):
                        end_f = mid_f
                    else:
                        start_f = mid_f + 1
                high_req_queue.insert(start_f, (f, arr))
            elif f in self.low_func:
                start_f, end_f = 0, len(low_req_queue)
                while start_f < end_f:
                    mid_f = (start_f + end_f) // 2
                    if calc_metric(low_req_queue[mid_f][0]) >= calc_metric(f):
                        end_f = mid_f
                    else:
                        start_f = mid_f + 1
                low_req_queue.insert(start_f, (f, arr))
            else:
                logging.warn('Warn: no func priority found')

        def get_req(out=True):
            # get first avail func request
            idx, req = None, None
            for i in range(len(high_req_queue)):
                if self.func_avail[high_req_queue[i][0]]:
                    idx, req = i, high_req_queue[i]
                    break
            if idx is not None:
                if out:
                    high_req_queue.pop(idx)
                return req
            for i in range(len(low_req_queue)):
                if self.func_avail[low_req_queue[i][0]]:
                    idx, req = i, low_req_queue[i]
                    break
            if idx is not None:
                if out:
                    low_req_queue.pop(idx)
                return req
            return None

            # if len(high_req_queue) > 0:
            #     return high_req_queue.pop(0)
            # return low_req_queue.pop(0)
        
        def batch_check(f):
            req = get_req(False)
            if req is None or req[0] != f:
                return False
            return True

        alpha_adj_window = 20
        alpha_adj_cur = 0
        old_slo_ratio = 0
        diff_threshold = 0.04
        scale_factor = 2.0
        low_scale_factor = 1.01

        alpha = 1

        logging.info('Listen on schedule queue using SLO-aware policy')

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
                    req = SignalRequest()
                    req.type = Execute
                    req.function = str(f)
                    server_id = self.send_signal(req, self.sockets)

                    issue_t = time.time()
                    self.func_avail[f] = False
                    self.sockets.get(get_manager_ipc(server_id), zmq.PUSH).send_pyobj(('1', (f, issue_t, arrs)))
                    logging.info(f'Scheduled request {f} to server {server_id}')

            else:
                if sched_typ == 0: # requests
                    target_func, arr_t = sched_msg
                    insert_req(target_func, arr_t)
                    logging.info(f'Inserted request {target_func}, cur_len: {len(high_req_queue) + len(low_req_queue)}')
                elif sched_typ == 1: # update queue

                    update_start_t = time.time()
                    alpha_adj_cur += 1
                    expected_count = [(f, (stat[2] / stat[1]) * (slo_percentile * stat[1] - stat[0]) / (1 - slo_percentile)) for f, stat in self.func_stat.items() if stat[1] > 0]
                    expected_count.sort(key=lambda x: x[1])

                    total_count = sum([i[1] for i in expected_count if i[1] >= 0]) * alpha
                    borderline, agg_count = -1, 0
                    for i in range(len(expected_count)):
                        if expected_count[i][1] > 0:
                            agg_count += expected_count[i][1]
                            if agg_count > total_count:
                                borderline = i - 1
                                break
                    if agg_count <= total_count:
                        borderline = len(expected_count) - 1

                    for i in range(len(expected_count)):
                        if i <= borderline:
                            self.high_func.add(expected_count[i][0])
                            self.low_func.discard(expected_count[i][0])
                        else:
                            self.high_func.discard(expected_count[i][0])
                            self.low_func.add(expected_count[i][0])

                    queue_req_count = len(high_req_queue) + len(low_req_queue)
                    old_high_req_queue, old_low_req_queue = high_req_queue.copy(), low_req_queue.copy()
                    high_req_queue.clear()
                    low_req_queue.clear()
                    for i in range(len(old_high_req_queue)):
                        insert_req(old_high_req_queue[i][0], old_high_req_queue[i][1])
                    for i in range(len(old_low_req_queue)):
                        insert_req(old_low_req_queue[i][0], old_low_req_queue[i][1])
                    
                    count = len([f for f, stat in self.func_stat.items() if stat[1] > 0])
                    if count <= 0:
                        continue

                    slo_ratio = len([f for f, stat in self.func_stat.items() if stat[1] > 0 and stat[0] / stat[1] >= slo_percentile]) / count
                    # func_slo_ratio.append((slo_ratio, alpha))
                    logging.info(f'Update queue. slo_ratio: {slo_ratio}, alpha: {alpha}, queued_req: {queue_req_count}, elasped: {time.time() - update_start_t}')

                    if alpha_adj_cur >= alpha_adj_window:
                        slo_ratio_diff = (slo_ratio - old_slo_ratio) / old_slo_ratio  if old_slo_ratio > 0 else diff_threshold
                        if slo_ratio_diff >= diff_threshold:
                            alpha = min(1, alpha * scale_factor)
                            old_slo_ratio = slo_ratio
                        elif slo_ratio_diff < - diff_threshold:
                            alpha = alpha / scale_factor
                            old_slo_ratio = slo_ratio
                        elif slo_ratio_diff >= 0 and slo_ratio_diff < diff_threshold:
                            alpha = min(1, alpha * low_scale_factor)
                        elif slo_ratio_diff >= - diff_threshold and slo_ratio_diff < 0:
                            alpha = min(1, alpha * low_scale_factor)

                        alpha_adj_cur = 0
                        logging.info(f'Update alpha. high func: {len(self.high_func)}, low func: {len(self.low_func)}, high queue {len(high_req_queue)}, low queue {len(low_req_queue)}, slo ratio {slo_ratio}, alpha {alpha}')


    def launch(self, func_id):
        req = SignalRequest()
        req.type = ExecuteAfterLoad
        req.function = str(func_id)
        req.payload = str(get_mixed_model_level(func_id))
        server_id = self.send_signal(req, self.sockets)
        self.sockets.get(get_manager_ipc(server_id), zmq.PUSH).send_pyobj(('0', func_id))
        
    def send_signal(self, req, sockets):
        signal_socket = sockets.get('ipc:///dev/shm/ipc/signal_0')
        signal_socket.send(req.SerializeToString())
        resp = SignalAck()
        resp.ParseFromString(signal_socket.recv())
        return resp.resp

    def test(self, func, gpu_id):
        req = SignalRequest()
        req.type = Execute
        req.function = str(func)
        req.payload = str(gpu_id)
        server_id = self.send_signal(req, self.sockets)

        req_start_t = time.time()
        x = requests.get('http://localhost:' + str(9000 + func), headers={"cur_server" : str(server_id), "batch_size": str(1)})
        req_end_t = time.time()
        return x.text
    
    def force_evict(self, func):
        req = SignalRequest()
        req.type = Unload
        req.function = str(func)
        self.send_signal(req, self.sockets)
    
    def run(self):
        self.pool.submit(self.poll_exterel)
        self.route_policy()

    def poll_exterel(self):
        global bert_function

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
                elif target_func == -1:
                    # clear up and warm up
                    clear_start_t = time.time()
                    if model_name == 'mixed':
                        for i in range(len(model_mixed)):
                            self.force_evict(i)
                    else:
                        self.force_evict(0)
                    logging.info(f'Clear up done')
                    for i in range(server_num):
                        if model_name == 'mixed':
                            for j in range(len(model_mixed)):
                                logging.info(f'Warm-up server {i} func {j} result {self.test(j, i)}')
                            time.sleep(0.1)
                            
                        else:
                            logging.info(f'Warm-up server {i} result {self.test(0, i)}')
                            time.sleep(1)
                    # if model_name == 'mixed':
                    #     for i in range(len(model_mixed)):
                    #         self.force_evict(i)
                    # else:
                    #     self.force_evict(0)
                    logging.info(f'Clear up and warm up done, time: {time.time() - clear_start_t}')
                elif target_func == -2:
                    # load all
                    for i in range(self.func_num):
                        req = SignalRequest()
                        req.type = Load
                        req.function = str(i)
                        self.send_signal(req, self.sockets)
                    logging.info(f'Load all func signal sent')


            if self.router_socket in socks and socks[self.router_socket] == zmq.POLLIN:
                obj = self.router_socket.recv_pyobj()
                flag, sched_msg = obj

                if flag == '0':
                    func_id = sched_msg
                    self.func_num += 1
                    self.func_stat[func_id] = [0, 0, 0]
                    self.high_func.add(func_id)
                    self.func_avail[func_id] = True

                elif flag == '1':
                    func_id, lats = sched_msg
                    ddl = bert_ddl if func_id in bert_function else latency_ddl
                    for lat in lats:
                        self.func_stat[func_id][2] += lat
                        self.func_stat[func_id][1] += 1
                        self.func_stat[func_id][0] = self.func_stat[func_id][0] + 1 if lat <= ddl else self.func_stat[func_id][0]
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
parser.add_argument('-l', type=int, default=0)
parser.add_argument('-p', type=str, default='sa')

args = parser.parse_args()

model_name = args.m
server_num = args.s     # GPU num
func_num = args.f       # Function num
slo_percentile = args.t / 100   # SLO percentile
latency_ddl = args.d / 1000 # Latency DDL
batch_size_limit = args.b   # Batch size limit
perform_launch = args.l < func_num
is_sa_policy = args.p == 'sa'

logging.info(f'Test args: model: {model_name}, server_num: {server_num}, func_num: {func_num}, slo_percentile: {slo_percentile}, latency_ddl: {latency_ddl}, batch_size_limit: {batch_size_limit}, perform_launch: {perform_launch}')

if perform_launch:
    router = Router(server_num, func_num=args.l, sa_policy=is_sa_policy)
    for i in range(args.l, func_num):
        router.launch(i)
else:
    router = Router(server_num, func_num=func_num, sa_policy=is_sa_policy)
router.run()

