import numpy as np
from datetime import datetime
import argparse

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

def parse_cross_lb(path):
    func_lats, _, _ = parse_log(path, parse_stamp=True, include_server=True)

    if not func_lats:
        print("No function call data was parsed from the log; the format may be incorrect or the log may be empty.")
        return

    start_t = min([info[0][0] for info in func_lats.values()])
    end_t = max([info[-1][0] for info in func_lats.values()])
    total_duration = end_t - start_t

    server_stats = [[0, [], []] for _ in range(4)]

    for infos in func_lats.values():
        for stamp, lats in infos:
            serv = lats[-1]
            e2e = lats[0]
            inf = lats[-2]
            server_stats[serv][0] += inf
            server_stats[serv][1].append(e2e)
            server_stats[serv][2].append(inf)

    indexed_stats = list(enumerate(server_stats))
    indexed_stats.sort(key=lambda x: x[1][0])

    print(f"\n{'GPU':<5}{'Normalized Load':>20}{'P98 Latency (ms)':>25}")
    print("-" * 50)
    for gpu_id, (total_inf, e2e_list, _) in indexed_stats:
        if len(e2e_list) == 0:
            print(f"{gpu_id:<5}{'N/A':>20}{'No data':>25}")
            continue
        norm_load = total_inf / total_duration
        p98_lat = np.percentile(e2e_list, 98) * 1000
        print(f"{gpu_id:<5}{norm_load:>20.4f}{p98_lat:>25.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse log file and print normalized load and P98 latency per GPU.")
    parser.add_argument("log_path", type=str, help="Path to the log file, e.g., logs/run1.log")
    args = parser.parse_args()

    parse_cross_lb(args.log_path)