import numpy as np
from datetime import datetime
import sys
import os

def parse_log(path, parse_stamp=False, include_server=False):
    func_lats = {}
    exclude_count = 0
    ignore_count = 0
    cur_lat_count = 0

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

    return func_lats

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <log_file_path>")
        sys.exit(1)

    log_path = sys.argv[1]

    if not os.path.isfile(log_path):
        print(f"File does not exist: {log_path}")
        sys.exit(1)

    print(f"Analyzing log file: {log_path}")
    func_lats = parse_log(log_path, parse_stamp=True)

    all_e2e = []
    all_timestamps = []

    for values in func_lats.values():
        for record in values:
            stamp = record[0]
            e2e = record[1][0]
            all_timestamps.append(stamp)
            all_e2e.append(e2e)

    if len(all_e2e) == 0 or len(all_timestamps) < 2:
        print("Insufficient logging for statistical purposes")
    else:
        p98 = np.percentile(all_e2e, 98)
        duration_min = (max(all_timestamps) - min(all_timestamps)) / 60
        throughput = len(all_e2e) / duration_min

        print(f"P98 latency: {p98 * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} requests/min")
        print(f"Total requests: {len(all_e2e)}, Running time: {duration_min:.2f} min")
