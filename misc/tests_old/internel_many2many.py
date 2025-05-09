import time
import sys
import zmq
from signal_pb2 import *
import subprocess
import requests

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


context = zmq.Context(1)
socket_cache = SocketCache(context)

def send_signal(req, server_id):
    signal_socket = socket_cache.get('ipc:///cuda/signal_' + str(server_id))
    signal_socket.send(req.SerializeToString())
    resp = SignalAck()
    resp.ParseFromString(signal_socket.recv())
    return resp

update_socket = context.socket(zmq.PULL)
update_socket.bind('ipc:///cuda/scheduler')

server_num = 1
if (len(sys.argv) > 1):
    server_num = int(sys.argv[1])

def send_launch_signal(func, server_id):
    req = SignalRequest()
    req.type = ExecuteAfterLoad
    req.function = str(func)
    req.payload = str(server_id)
    for i in range(server_num):
        ack = send_signal(req, i)

func_server_map = {}
def update_func_server_map(func, cur_server):
    if func not in func_server_map:
        func_server_map[func] = [cur_server]
        print(f'Add server {cur_server} to func {func} server list')
    else:
        if cur_server not in func_server_map[func]:
            func_server_map[func].append(cur_server)
            print(f'Add server {cur_server} to func {func} server list')

def launch_client(client_id, server_id):
    send_launch_signal(client_id, server_id)
    print(f'ExecuteAfterLoad {client_id} signal')

    p = subprocess.Popen(["bash", "/start_with_server_id.sh", str(client_id), str(server_id), "cv_endpoint.py", str(9000 + client_id)])
    time.sleep(10)
    
    x = requests.get('http://localhost:' + str(9000 + client_id), headers={"cur_server" : str(server_id)})
    print(f'ExecuteAfterLoad {client_id} on server {server_id} resp: {x.text}')
    time.sleep(1)
    update_func_server_map(client_id, server_id)

    req = SignalRequest()
    req.type = Unload
    req.function = str(client_id)
    ack = send_signal(req, server_id)
    print(f'Unload {client_id} on server {server_id} signal ack {ack.ack}')


def load_model(func, server_id):
    if func in func_server_map:
        if server_id in func_server_map[func]:
            return
        source_server_id = func_server_map[func][0]
        
        req = SignalRequest()
        req.type = Load
        req.function = str(func)
        req.payload = str(source_server_id)

        ack = send_signal(req, server_id)
        print(f'Issue load {func} on server {server_id} signal ack {ack}')

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

poller = zmq.Poller()
poller.register(update_socket, zmq.POLLIN)

for i in range(server_num):
    launch_client(i, i)

for i in range(server_num):
    for j in range(server_num):
        load_model(i, j)

while True:
    socks = dict(poller.poll(timeout=100))
    if update_socket in socks and socks[update_socket] == zmq.POLLIN:
        msg = UpdateServerStatus()
        msg.ParseFromString(update_socket.recv())

        cur_server = msg.server_id
        functions = [int(func) for func in msg.functions]
        for func in functions:
            update_func_server_map(func, cur_server)
    
    if all([len(ls) == server_num for func, ls in func_server_map.items()]):
        break
    
print(f'Finish load')

for _ in range(2):
    for i in range(server_num):
        for j in range(server_num):
            execute(j, i)

