import numpy as np
from datetime import datetime
import argparse
import glob
import matplotlib.pyplot as plt
import pandas as pd
import math
from pylab import *

def parse_log(path, parse_stamp=False, include_server=False):
    func_lats = {}
    alpha_ratio = []
    exclude_count = 0
    req_counts = []
    ignore_count = 0
    cur_lat_count, cur_req_count = 0, 0
    with open(path, "r") as f:
        for line in f:
            if 'end-to-end time' in line:
                if cur_lat_count < ignore_count:
                    cur_lat_count += 1
                    continue
                begin_func = line.find('Func') + 5
                end_func = line.find(' ', begin_func)
                func = int(line[begin_func:end_func])
                
                begin_e2e = line.find('end-to-end time') + 17
                end_e2e = line.find(',', begin_e2e)
                try:
                    e2e = float(line[begin_e2e:end_e2e])
                except:
                    exclude_count += 1
                    continue

                begin_qry = line.find('query') + 7
                end_qry = line.find(',', begin_qry)
                try:
                    qry = float(line[begin_qry:end_qry])
                except:
                    exclude_count += 1
                    continue
                    
                begin_inf = line.find('elasped') + 9
                end_inf = line.find('"', begin_inf)
                try:
                    inf = float(line[begin_inf:end_inf])
                except:
                    exclude_count += 1
                    continue
                    
                info = [e2e, qry, inf]
                if 'issue' in line:
                    begin_issue = line.find('issue') + 7
                    end_issue = line.find(',', begin_issue)
                    issue = float(line[begin_issue:end_issue])
                    info = [e2e, issue, qry, inf]

                if include_server:
                    begin_svr = line.find('server') + 7
                    svr = int(line[begin_svr:begin_svr+1])
                    info.append(svr)
                    
                if parse_stamp:
                    dt_str = line[:23]
                    dt = datetime.strptime(dt_str, "%Y-%m-%d,%H:%M:%S.%f")
                    stamp = datetime.timestamp(dt)
                    info = (stamp, info)
                    
                if func in func_lats:
                    func_lats[func].append(info)
                else:
                    func_lats[func] = [info]
                    
            if 'Update queue' in line:
                begin_ratio = line.find('ratio') + 7
                end_ratio = line.find(',', begin_ratio)
                ratio = float(line[begin_ratio:end_ratio])
                
                begin_alpha = line.find('alpha') + 7
                end_alpha = line.find(',', begin_alpha)
                alpha = float(line[begin_alpha:end_alpha])
                alpha_ratio.append((alpha, ratio))

            
            if 'FIFO update' in line:
                begin_ratio = line.find('ratio') + 7
                end_ratio = line.find(',', begin_ratio)
                ratio = float(line[begin_ratio:end_ratio])
                
                alpha_ratio.append((0, ratio))


            if 'Inserted request' in line:
                if cur_req_count < ignore_count:
                    cur_req_count += 1
                    continue
                begin_req = line.find('cur_len') + 9
                req_count = int(line[begin_req:])
                
                req_counts.append(req_count)
            
    return func_lats, alpha_ratio, req_counts


def compare_policy(path, ddl, p=0.98):
    def times100(x):
        return np.array(x) * 100
    
    linestyles = ['-', '--', ':', '-.', '--']
    markers = ['o', 'v', '^', 's', 'D']
    system_name = 'Torpor'
    to_plot = {}
    for sub in glob.glob(f'{path}/*'):
        name = sub.split('/')[-1].split('_')[1]
        count = sub.split('/')[-1].split('_')[0]
        func_lats, alpha_ratio, req_counts = parse_log(f'{sub}/router.log')
        
        slo_meet = [ [0,0] for i in range(8)]
        for f in range(len(func_lats)):
            lats = func_lats[f]
            func_idx = f % 8
            d = 0.2 if func_idx == 7 else ddl
            slo_meet[func_idx] = [slo_meet[func_idx][0] + 1 if len([l for l in lats if l[0] <= d]) / len(lats) >= p else slo_meet[func_idx][0], slo_meet[func_idx][1] + 1]
        total_ratio = sum([i[0] for i in slo_meet]) / sum([i[1] for i in slo_meet])
        to_plot[name] = to_plot[name] + [(count, total_ratio)] if name in to_plot else [(count, total_ratio)]
    
    fig = plt.figure(figsize=(8, 3.2))
    ax = fig.add_subplot(111)
    name_plot = [('full', system_name), ('fifo', f'{system_name}-FIFO'), ('fixblock', f'{system_name}-Block'), ('lru', f'{system_name}-LRU'), ('rand', f'{system_name}-Random')]
    for i in range(len(name_plot)):
        res = to_plot[name_plot[i][0]]
        res.sort(key=lambda e: e[0])
        x = [j[0] for j in res]
        d = [j[1] for j in res]
        
        ax.plot(x, times100(d), linewidth=3, linestyle=linestyles[i], marker=markers[i], markersize=10, label=name_plot[i][1])

    ax.set_xlabel('Number of functions')
    ax.set_ylabel('SLO-compliant Func.(%)')
    ax.yaxis.set_label_coords(-0.1, 0.43)    
    ax.grid(True)
    ax.legend(loc='lower center')

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse log and compute SLO compliance.")
    parser.add_argument("log_path", type=str, help="Path to the log dir, which can be downloaded from the shared Google Drive link.")
    args = parser.parse_args()

    compare_policy(args.log_path, 0.08, p=.98)