import time
import sys
import os
import zmq
import numpy as np

context = zmq.Context(1)
sender_socket = context.socket(zmq.PUSH)
sender_socket.connect('ipc:///dev/shm/ipc/externel_service')
data_path = '../tools'

class Sender:
    def __init__(self, func_num, rate, mins):
        self.func_num = func_num
        if rate <= 0:
            self.pattern = []
            with open(f'{data_path}/invoc_pattern.txt', "r") as f:
                for line in f:
                    self.pattern += [float(x.strip()) for x in line.split(',')]
            print(f'Sender pattern size {len(self.pattern)}, avg rpm {np.average(self.pattern)}')
            
            # self.func_rate = np.random.choice(self.pattern, func_num)
            # np.random.shuffle(self.func_rate)

            # test fixed rate
            self.func_rate = [np.average(self.pattern)] * func_num
        else:
            self.func_rate = [rate] * func_num

        if os.path.isfile(f'{data_path}/req_arrivals_{func_num}_{rate}_{mins}.npy'):
            self.req_arrivals = np.load(f'{data_path}/req_arrivals_{func_num}_{mins}.npy')
        else:
            self.req_arrivals = []
            self.gen_req_arrivals(self.req_arrivals, mins = mins)

            with open(f'{data_path}/req_arrivals_{func_num}_{mins}.npy', 'wb') as f:
                np.save(f, self.req_arrivals)
        print(f'Generated {len(self.req_arrivals)} requests in {mins} mins')

        # print(f'{self.req_arrivals[:10]}')

    def send(self, func):
        sender_socket.send_string(str(func))
        # print(f'Send func {func}')

    def gen_req_arrivals(self, req_arrivals, mins = 5):
        cur_stmp = 0
        overall_rate = int(sum(self.func_rate))

        for _ in range(mins):
            intervals = np.random.poisson(60 * 1000 / overall_rate, overall_rate)
            funcs = np.random.choice(len(self.func_rate), overall_rate, p = [r / sum(self.func_rate) for r in self.func_rate])

            for i, f in zip(intervals, funcs):
                cur_stmp += i / 1000
                req_arrivals.append((i / 1000, f))

    
    def run(self, flag=0):
        if flag == -1:
            # load
            self.send(-2)
        else:
            if flag == 0: # warmup
                self.send(-1)
                print(f'Warmup and sleep {0.05 * func_num + 5} sec')
                time.sleep(0.05 * func_num + 5)

            print('Start sending requests')
            for i, f in self.req_arrivals:
                func = int(f)
                # print(f'Send func {func} after sleep {i}')
                time.sleep(i)
                self.send(func)
                    
def test(concurrency):
    for i in range(concurrency):
        sender_socket.send_string(str(i))
        sender_socket.send_string(str(i))

if __name__ == '__main__':
    func_num = 1
    if (len(sys.argv) > 1):
        func_num = int(sys.argv[1])

    rate = 0
    if (len(sys.argv) > 2):
        rate = int(sys.argv[2])

    mins = 5
    if (len(sys.argv) > 3):
        mins = int(sys.argv[3])

    flag = 0
    if (len(sys.argv) > 4):
        flag = int(sys.argv[4])
    sender = Sender(func_num, rate, mins)
    sender.run(flag)


