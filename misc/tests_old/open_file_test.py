import time
import sys
import os
import zmq

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


if __name__ == '__main__':
    f_num = 1
    if (len(sys.argv) > 1):
        f_num = int(sys.argv[1])

    context = zmq.Context(1)
    context.set(zmq.MAX_SOCKETS, 100000)
    cache = SocketCache(context)
    for i in range(f_num):
        cache.get(f'ipc:///dev/shm/ipc/client_{i}')
