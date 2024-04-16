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
    def __init__(self, func_num, mins):
        self.func_num = func_num
        self.req_arrivals = []
        if os.path.isfile(f'{data_path}/req_arrivals_{func_num}_trace.npy'):
            self.req_arrivals = np.load(f'{data_path}/req_arrivals_{func_num}_trace.npy')
        else:
            self.pattern = np.load(f'{data_path}/samples.npy')
            self.func_rate = [ self.pattern[i] for i in np.random.choice(list(range(len(self.pattern))), func_num)]
            print(f'Sender sample size {len(self.pattern)}, min len {len(self.pattern[0])}')

            cur_stmp = 0
            for rate_list in zip(*self.func_rate):
                overall_rate = int(sum(rate_list))

                intervals = np.random.poisson(60 * 1000 / overall_rate, overall_rate)
                funcs = np.random.choice(len(rate_list), overall_rate, p = [r / sum(rate_list) for r in rate_list])

                for i, f in zip(intervals, funcs):
                    cur_stmp += i / 1000
                    self.req_arrivals.append((i / 1000, f))


        with open(f'{data_path}/req_arrivals_{func_num}_trace.npy', 'wb') as f:
            np.save(f, self.req_arrivals)

        print(f'Generated {len(self.req_arrivals)} requests from trace')

    def send(self, func):
        sender_socket.send_string(str(func))
        # print(f'Send func {func}')
    
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
    
    mins = 5
    if (len(sys.argv) > 2):
        mins = int(sys.argv[2])

    flag = 0
    if (len(sys.argv) > 3):
        flag = int(sys.argv[3])
    sender = Sender(func_num, mins)
    sender.run(flag)


