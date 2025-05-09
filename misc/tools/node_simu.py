# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
import math
import os
import sys
import collections
import random
import argparse
 
class LRUCache:
  def __init__(self, size):
    self.size = size
    self.lru_cache = collections.OrderedDict()
    self.evict_count = 0
 
  def has(self, key):
    return key in self.lru_cache
 
  def add(self, key):
    if key in self.lru_cache:
        self.lru_cache.pop(key)
    else:
        if len(self.lru_cache) >= self.size:
            self.lru_cache.popitem(last=False)
            self.evict_count += 1
    self.lru_cache[key] = 0

def get_req_arrivals_old(func_num, mins = 5):
    req_arrivals = []
    if os.path.isfile(f'req_arrivals_{func_num}_{mins}.npy'):
        req_arrivals = np.load(f'req_arrivals_{func_num}_{mins}.npy')
    else:
        pattern = []
        with open("invoc_pattern.txt", "r") as f:
            for line in f:
                pattern += [float(x.strip()) for x in line.split(',')]
        print(f'Sender pattern size {len(pattern)}, avg rpm {np.average(pattern)}')
        
        # func_rate = np.random.choice(pattern, func_num)
        # np.random.shuffle(func_rate)
        func_rate = [16.5] * func_num

        def gen_arrivals():
            cur_stmp = 0
            overall_rate = int(sum(func_rate))

            for _ in range(mins):
                intervals = np.random.poisson(60 * 1000 / overall_rate, overall_rate)
                funcs = np.random.choice(len(func_rate), overall_rate, p = [r / sum(func_rate) for r in func_rate])

                for i, f in zip(intervals, funcs):
                    cur_stmp += i / 1000
                    req_arrivals.append((i / 1000, f))


        req_arrivals = []
        gen_arrivals()

        with open(f'req_arrivals_{func_num}_{mins}.npy', 'wb') as f:
            np.save(f, req_arrivals)

    print(f'Generated {len(req_arrivals)} requests in {mins} mins')
    return req_arrivals

def get_req_arrivals(func_num, mins = 5):
    req_arrivals = []
    if os.path.isfile(f'req_arrivals_{func_num}_trace.npy'):
        req_arrivals = np.load(f'req_arrivals_{func_num}_trace.npy')
    else:
        pattern = np.load("samples.npy")
        func_rate = [ pattern[i] for i in np.random.choice(list(range(len(pattern))), func_num)]
        print(f'Sender sample size {len(pattern)}, min len {len(pattern[0])}')


        def gen_arrivals():
            cur_stmp = 0
            for rate_list in zip(*func_rate):
                overall_rate = int(sum(rate_list))

                intervals = np.random.poisson(60 * 1000 / overall_rate, overall_rate)
                funcs = np.random.choice(len(rate_list), overall_rate, p = [r / sum(rate_list) for r in rate_list])

                for i, f in zip(intervals, funcs):
                    cur_stmp += i / 1000
                    req_arrivals.append((i / 1000, f))


        req_arrivals = []
        gen_arrivals()

        with open(f'req_arrivals_{func_num}_trace.npy', 'wb') as f:
            np.save(f, req_arrivals)

    print(f'Generated {len(req_arrivals)} requests in {mins} mins')
    return req_arrivals

a_noswap_lat, a_g2g_lat, a_h2g_lat, a_h2g_lat_con, a_ddl, a_size = 0.019, 0.019, 0.025, 0.041, 0.05, 0.3
b_noswap_lat, b_g2g_lat, b_h2g_lat, b_h2g_lat_con, b_ddl, b_size = 0.015, 0.015, 0.02, 0.033, 0.05, 0.25

def get_func_metadata(func_id):
    return a_noswap_lat, a_g2g_lat, a_h2g_lat, a_h2g_lat_con, a_ddl, a_size
    # return b_noswap_lat, b_g2g_lat, b_h2g_lat, b_h2g_lat_con, b_ddl, b_size

gpu_pairs = [(0, 1), (2, 3)]
def get_cohenrence_gpu(gpu_id):
    for p in gpu_pairs:
        if gpu_id in p:
            for i in p:
                if i != gpu_id:
                    return i


SWAP_FLAG_None, SWAP_FLAG_G2G, SWAP_FLAG_H2G, SWAP_FLAG_H2G_CON = 0, 1, 2, 3
gpu_num = 4
size_per_gpu = 30

old_slo_ratio = 0
def simulation(func_num, req_arrivals):
    cache_sz = int(size_per_gpu // get_func_metadata(0)[-1])
    print(f'Cache size {cache_sz} per GPU')
    co_req_count = [0 for _ in range(gpu_num)]
    gpu_util = [0 for _ in range(gpu_num)]
    gpu_func_map = [LRUCache(cache_sz) for _ in range(gpu_num)]
    gpu_last_times = [[-1, None, None, SWAP_FLAG_None] for _ in range(gpu_num)] # [last issue time, func_id, arrival_time, last swappness]
    func_stat = [[0, 0] for _ in range(func_num)] # [SLO compliance, total]
    period_func_stat = [[0, 0] for _ in range(func_num)]
    func_lat = []
    func_slo_ratio = []
    total_req = 0

    # # naive
    # def schedule(func_id, idle_gpus):
    #     for i in idle_gpus:
    #         if gpu_func_map[i].has(func_id):
    #             return i, SWAP_FLAG_None
    #     return random.choice(idle_gpus), SWAP_FLAG_H2G

    # best-effort
    def schedule(func_id, idle_gpus):
        for i in idle_gpus:
            if gpu_func_map[i].has(func_id):
                return i, SWAP_FLAG_None
        gid = np.argmin([gpu_util[i] for i in idle_gpus])

        for i in range(gpu_num):
            if gpu_func_map[i].has(func_id):
                return idle_gpus[gid], SWAP_FLAG_G2G
        
        return idle_gpus[gid], SWAP_FLAG_H2G

    def try_update(func_stat):
        pass

    # # single queue
    # req_queue = []
    # def has_req():
    #     return len(req_queue) > 0
    # def insert_req(f, arr):
    #     req_queue.append((f, arr))
    # # def insert_req(f, arr):
    # #     if func_stat[f][1] == 0:
    # #         req_queue.insert(0, (f, arr))
    # #         return

    # #     start_f, end_f = 0, len(req_queue)
    # #     while start_f < end_f:
    # #         mid_f = (start_f + end_f) // 2
    # #         if func_stat[req_queue[mid_f][0]][1] > 0 and func_stat[req_queue[mid_f][0]][0] / func_stat[req_queue[mid_f][0]][1] > func_stat[f][0] / func_stat[f][1]:
    # #             end_f = mid_f
    # #         else:
    # #             start_f = mid_f + 1
    # #     req_queue.insert(start_f, (f, arr))

    # def get_req():
    #     return req_queue.pop(0)

    # two priority queue
    high_req_queue, low_req_queue = [], []
    high_func, low_func = set([i for i in range(func_num)]), set([])
    def has_req():
        return len(high_req_queue) > 0 or len(low_req_queue) > 0

    window_func = 400
    window_alpha = window_func * scale_window

    def try_update(func_stat):
        if total_req > 0 and total_req % window_func == 0:
            global alpha
            global old_slo_ratio
            expected_count = [(i, (ratio_threshold * func_stat[i][1] - func_stat[i][0]) / (1 - ratio_threshold)) for i in range(func_num)]
            expected_count.sort(key=lambda x: x[1])

            if alpha >= 0:
                total_count = sum([i[1] for i in expected_count if i[1] > 0]) * alpha
                borderline, agg_count = -1, 0
                for i in range(len(expected_count)):
                    if expected_count[i][1] > 0:
                        agg_count += expected_count[i][1]
                        if agg_count > total_count:
                            borderline = i - 1
                            break
                if agg_count <= total_count:
                    borderline = len(expected_count) - 1
            else:
                total_count = sum([i[1] for i in expected_count if i[1] < 0]) * (-alpha)
                borderline, agg_count = -1, 0
                for i in range(len(expected_count) - 1, -1, -1):
                    if expected_count[i][1] < 0:
                        agg_count += -expected_count[i][1]
                        if agg_count > total_count:
                            borderline = i
                            break
                if agg_count <= total_count:
                    borderline = -1
            for i in range(len(expected_count)):
                if i <= borderline:
                    high_func.add(expected_count[i][0])
                    low_func.discard(expected_count[i][0])
                else:
                    high_func.discard(expected_count[i][0])
                    low_func.add(expected_count[i][0])

            old_high_req_queue, old_low_req_queue = high_req_queue.copy(), low_req_queue.copy()
            high_req_queue.clear()
            low_req_queue.clear()
            for i in range(len(old_high_req_queue)):
                insert_req(old_high_req_queue[i][0], old_high_req_queue[i][1])
            for i in range(len(old_low_req_queue)):
                insert_req(old_low_req_queue[i][0], old_low_req_queue[i][1])
            
            slo_ratio = len([i for i in range(func_num) if func_stat[i][1] > 0 and func_stat[i][0] / func_stat[i][1] >= ratio_threshold]) / len([i for i in range(func_num) if func_stat[i][1] > 0])
            # slo_ratio = len([i for i in range(func_num) if func_stat[i][1] > 0 and func_stat[i][0] / func_stat[i][1] >= ratio_threshold]) / func_num
            func_slo_ratio.append((slo_ratio, alpha))

            # if total_req % window_alpha == 0:
            #     slo_ratio_diff = (slo_ratio - old_slo_ratio) / old_slo_ratio  if old_slo_ratio > 0 else diff_threshold
            #     if slo_ratio_diff >= diff_threshold:
            #         alpha = min(1, alpha * scale_factor)
            #         old_slo_ratio = slo_ratio
            #     elif slo_ratio_diff < - diff_threshold:
            #         alpha = alpha / scale_factor
            #         old_slo_ratio = slo_ratio
            #     elif slo_ratio_diff >= 0 and slo_ratio_diff < diff_threshold:
            #         alpha = min(1, alpha * low_scale_factor)
            #     elif slo_ratio_diff >= - diff_threshold and slo_ratio_diff < 0:
            #         alpha = min(1, alpha * low_scale_factor)

            #     # if slo_ratio > old_slo_ratio:
            #     #     alpha = min(1, alpha * scale_factor)
            #     # elif slo_ratio < old_slo_ratio:
            #     #     alpha = alpha / scale_factor
            #     # old_slo_ratio = slo_ratio

            #     print(f'High func: {len(high_func)}, low func: {len(low_func)}, high queue {len(high_req_queue)}, low queue {len(low_req_queue)}, slo ratio {slo_ratio}, alpha {alpha}')

    def insert_req(f, arr):
        if f in high_func:
            start_f, end_f = 0, len(high_req_queue)
            while start_f < end_f:
                mid_f = (start_f + end_f) // 2
                if ratio_threshold * func_stat[high_req_queue[mid_f][0]][1] - func_stat[high_req_queue[mid_f][0]][0] < ratio_threshold * func_stat[f][1] - func_stat[f][0]:
                    end_f = mid_f
                else:
                    start_f = mid_f + 1
            high_req_queue.insert(start_f, (f, arr))
        elif f in low_func:
            start_f, end_f = 0, len(low_req_queue)
            while start_f < end_f:
                mid_f = (start_f + end_f) // 2
                if ratio_threshold * func_stat[low_req_queue[mid_f][0]][1] - func_stat[low_req_queue[mid_f][0]][0] >= ratio_threshold * func_stat[f][1] - func_stat[f][0]:
                    end_f = mid_f
                else:
                    start_f = mid_f + 1
            low_req_queue.insert(start_f, (f, arr))
        else:
            print('Warn: no func priority found')

    # def insert_req(f, arr):
    #     def try_update_func_priority(f):
    #         if func_stat[f][1] > 0 and func_stat[f][1] % 80 == 0:
    #             if func_stat[f][0] / func_stat[f][1] >= ratio_threshold and f in low_func:
    #                 low_func.remove(f)
    #                 high_func.add(f)
    #             elif func_stat[f][0] / func_stat[f][1] < ratio_threshold and f in high_func:
    #                 high_func.remove(f)
    #                 low_func.add(f)
    #     if func_stat[f][1] == 0:
    #         high_req_queue.insert(0, (f, arr))
    #         return

    #     try_update_func_priority(f)
    #     if f in high_func:
    #         start_f, end_f = 0, len(high_req_queue)
    #         while start_f < end_f:
    #             mid_f = (start_f + end_f) // 2
    #             if func_stat[high_req_queue[mid_f][0]][1] > 0 and func_stat[high_req_queue[mid_f][0]][0] / func_stat[high_req_queue[mid_f][0]][1] > func_stat[f][0] / func_stat[f][1]:
    #                 end_f = mid_f
    #             else:
    #                 start_f = mid_f + 1
    #         high_req_queue.insert(start_f, (f, arr))
    #     elif f in low_func:
    #         start_f, end_f = 0, len(low_req_queue)
    #         while start_f < end_f:
    #             mid_f = (start_f + end_f) // 2
    #             if func_stat[low_req_queue[mid_f][0]][1] > 0 and func_stat[low_req_queue[mid_f][0]][0] / func_stat[low_req_queue[mid_f][0]][1] <= func_stat[f][0] / func_stat[f][1]:
    #                 end_f = mid_f
    #             else:
    #                 start_f = mid_f + 1
    #         low_req_queue.insert(start_f, (f, arr))
    #     else:
    #         print('Warn: no func priority found')

    def get_req():
        if len(high_req_queue) > 0:
            return high_req_queue.pop(0)
        return low_req_queue.pop(0)

    cur_time = 0
    for cur_i, cur_f in req_arrivals:
        total_req += 1
        try_update(func_stat)
        cur_f = int(cur_f)
        cur_time += cur_i

        expected_idle_time = [-1 if gpu_last_times[i][1] is None else gpu_last_times[i][0] + get_func_metadata(gpu_last_times[i][1])[gpu_last_times[i][-1]] for i in range(gpu_num)]
        first_idle = np.argmin(expected_idle_time) if min(expected_idle_time) <= cur_time else -1

        def handle_schedule_res(sched_t, arr_t, gid, func, flag):
            # record previous func meta
            if gpu_last_times[gid][1] is not None:
                f = gpu_last_times[gid][1]
                # print(f'f {f}')
                exec_time = expected_idle_time[gid] - gpu_last_times[gid][2]
                func_stat[f][0] = func_stat[f][0] if exec_time > get_func_metadata(f)[4] else func_stat[f][0] + 1
                func_stat[f][1] += 1

                period_func_stat[f][0] = period_func_stat[f][0] if exec_time > get_func_metadata(f)[4] else period_func_stat[f][0] + 1
                period_func_stat[f][1] += 1

                func_lat.append(exec_time)
                gpu_util[gid] += expected_idle_time[gid] - gpu_last_times[gid][0]

            if flag == SWAP_FLAG_H2G:
                co_gpu = get_cohenrence_gpu(gid)
                if gpu_last_times[co_gpu][-1] >= SWAP_FLAG_H2G and expected_idle_time[co_gpu] - sched_t >= 0.5 * get_func_metadata(func)[SWAP_FLAG_H2G]:
                    gpu_last_times[gid] = [sched_t, func, arr_t, SWAP_FLAG_H2G_CON]
                    expected_idle_time[gid] = sched_t + get_func_metadata(func)[SWAP_FLAG_H2G_CON]
                    co_req_count[gid] += 1

                    if gpu_last_times[co_gpu][-1] == SWAP_FLAG_H2G: # update co_gpu
                        gpu_last_times[co_gpu][-1] = SWAP_FLAG_H2G_CON
                        expected_idle_time[co_gpu] = expected_idle_time[co_gpu] + get_func_metadata(gpu_last_times[co_gpu][1])[SWAP_FLAG_H2G_CON] - get_func_metadata(gpu_last_times[co_gpu][1])[SWAP_FLAG_H2G]
                        co_req_count[co_gpu] += 1

                else:
                    gpu_last_times[gid] = [sched_t, func, arr_t, SWAP_FLAG_H2G]
                    expected_idle_time[gid] = sched_t + get_func_metadata(func)[SWAP_FLAG_H2G]

                gpu_func_map[gid].add(func)

            elif flag == SWAP_FLAG_None : # SWAP_FLAG_None
                gpu_last_times[gid] = [sched_t, func, arr_t, SWAP_FLAG_None]
                expected_idle_time[gid] = sched_t + get_func_metadata(func)[SWAP_FLAG_None]
            elif flag == SWAP_FLAG_G2G:
                gpu_last_times[gid] = [sched_t, func, arr_t, SWAP_FLAG_G2G]
                expected_idle_time[gid] = sched_t + get_func_metadata(func)[SWAP_FLAG_G2G]

                gpu_func_map[gid].add(func)
            else:
                print(f'Warn: unexpected swap flag {flag}')
                return

        while has_req() and first_idle >= 0:
            req = get_req()
            gpu_id, swap_flag = schedule(req[0], [first_idle])
            issue_time = expected_idle_time[gpu_id]

            handle_schedule_res(issue_time, req[1], gpu_id, req[0], swap_flag)
            first_idle = np.argmin(expected_idle_time) if min(expected_idle_time) <= cur_time else -1

        if first_idle < 0:
            insert_req(cur_f, cur_time)
            continue
        
        # execute current request
        idle_gpus = [i for i in range(gpu_num) if expected_idle_time[i] <= cur_time]
        gpu_id, swap_flag = schedule(cur_f, idle_gpus)
        handle_schedule_res(cur_time, cur_time, gpu_id, cur_f, swap_flag)


    while has_req():
        first_idle = np.argmin(expected_idle_time)
        req = get_req()
        gpu_id, swap_flag = schedule(req[0], [first_idle])
        issue_time = expected_idle_time[gpu_id]

        handle_schedule_res(issue_time, req[1], gpu_id, req[0], swap_flag)

    for i in range(gpu_num):
        f = gpu_last_times[i][1]
        exec_time = expected_idle_time[i] - gpu_last_times[i][2]
        func_stat[f][0] = func_stat[f][0] if exec_time > get_func_metadata(f)[4] else func_stat[f][0] + 1
        func_stat[f][1] += 1

        period_func_stat[f][0] = period_func_stat[f][0] if exec_time > get_func_metadata(f)[4] else period_func_stat[f][0] + 1
        period_func_stat[f][1] += 1

        func_lat.append(exec_time)

        gpu_util[i] += expected_idle_time[i] - gpu_last_times[i][0]
        print(f'gpu {i} util {gpu_util[i] / expected_idle_time[i]}')

    for i in range(gpu_num):
        print(f'gpu {i} evict_count {gpu_func_map[i].evict_count}')
    
    # print(f'high_func {len(high_func)}, low_func {len(low_func)}')
    print(f'co_req_count {sum(co_req_count)}')
    
    return func_stat, func_slo_ratio

parser = argparse.ArgumentParser()
parser.add_argument('-f', type=int, default=400)
parser.add_argument('-m', type=int, default=5)
parser.add_argument('-t', type=int, default=98)
parser.add_argument('-a', type=int, default=1)
parser.add_argument('-s', type=int, default=200)
parser.add_argument('-i', type=int, default=4)
parser.add_argument('-w', type=int, default=20)

args = parser.parse_args()

func_num = args.f
mins = args.m
ratio_threshold = args.t / 100
alpha = args.a / 100
scale_factor = args.s / 100
diff_threshold = args.i / 100
scale_window = args.w

low_scale_factor = 1.01

print(f'ratio_threshold {ratio_threshold}, alpha {alpha}, scale_factor {scale_factor}, diff_threshold {diff_threshold}')

req_arr = get_req_arrivals(func_num, mins)
func_stat, func_slo_ratio = simulation(func_num, req_arr)
total_req = sum([func_stat[i][1] for i in range(func_num)])
total_slo = sum([func_stat[i][0] for i in range(func_num)])
print(f'slo {total_slo} total {total_req}: {total_slo / total_req}')
slo_ratio = len([i for i in range(func_num) if func_stat[i][1] > 0 and func_stat[i][0] / func_stat[i][1] >= ratio_threshold]) / len([i for i in range(func_num) if func_stat[i][1] > 0])
print(f'slo ratio {slo_ratio}')
func_slo_ratio.append((slo_ratio, alpha))

func_slo = [func_stat[i][0] / func_stat[i][1] for i in range(func_num)]

with open(f'slo_{func_num}_{mins}.npy', 'wb') as f:
    np.save(f, func_slo)

with open(f'stat_{func_num}_{mins}.npy', 'wb') as f:
    np.save(f, func_stat)

with open(f'step_{func_num}_{mins}.npy', 'wb') as f:
    np.save(f, func_slo_ratio)

# func_slo.sort()
# print(func_slo[:10])
# for i in range(func_num):
#     print(f'{i}  {func_stat[i][0] / func_stat[i][1]}')