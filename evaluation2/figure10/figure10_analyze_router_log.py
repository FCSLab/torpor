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

def parse_cross_lb(path, total_function_count, slo_percentile=0.98, latency_ddl=0.1):
    func_lats, _, _ = parse_log(path, parse_stamp=True, include_server=True)

    if not func_lats:
        print("No function call data was parsed from the log; the format may be incorrect or the log may be empty.")
        return

    slo_compliant_count = 0
    actual_func_count = 0

    for func_id, records in func_lats.items():
        if len(records) < 1:
            continue

        actual_func_count += 1
        latencies = [entry[1][0] for entry in records]  # e2e latency in seconds
        within_ddl = [lat for lat in latencies if lat <= latency_ddl]

        ratio = len(within_ddl) / len(latencies)
        if ratio >= slo_percentile:
            slo_compliant_count += 1

    slo_compliance_percentage = slo_compliant_count / total_function_count * 100 if total_function_count > 0 else 0

    print("\n===== SLO Compliance Summary =====")
    print(f"Total function count         : {total_function_count}")
    print(f"Actual function count        : {actual_func_count}")
    print(f"SLO-compliant function count: {slo_compliant_count}")
    print(f"SLO-compliant Func. (%)     : {slo_compliance_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse log and compute SLO compliance.")
    parser.add_argument("log_path", type=str, help="Path to the log file, e.g., logs/run1.log")
    parser.add_argument("--total_func", type=int, required=True, help="Total number of deployed functions (e.g., 160)")
    args = parser.parse_args()

    parse_cross_lb(args.log_path, total_function_count=args.total_func, slo_percentile=0.98, latency_ddl=0.1)