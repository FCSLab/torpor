import subprocess
import time
import argparse
import zmq

def start_standalone_server():
    print("Starting standalone-server container...")

    docker_cmd = [
        "docker", "run", "--gpus", "all",
        "--rm",
        "--network=host",
        "--ipc=host",
        "-v", "/dev/shm/ipc:/cuda",
        "-e", "MEM_LIMIT_IN_GB=25",
        "-e", "IO_THREAD_NUM=4",
        "-dit",
        "--name", "standalone-server",
        "standalone-server",
        "bash", "start.sh"
    ]

    result = subprocess.run(docker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Failed to start container:", result.stderr)
        return None

    container_id = result.stdout.strip()
    print(f"Container started with ID: {container_id}")
    return container_id

def wait_for_server_ready(container_id, timeout=60):
    print("Waiting for server to be ready...")

    expected_lines = {
        "Server 0 -- Set I/O thread num to 4",
        "Server 1 -- Set I/O thread num to 4",
        "Server 2 -- Set I/O thread num to 4",
        "Server 3 -- Set I/O thread num to 4"
    }

    start_time = time.time()
    matched_lines = set()

    log_proc = subprocess.Popen(
        ["docker", "logs", "-f", container_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    try:
        for line in log_proc.stdout:
            line = line.strip()
            for expected in expected_lines:
                if expected in line:
                    matched_lines.add(expected)
                    print(f"Detected: {line}")
            if matched_lines == expected_lines:
                print("Server is fully ready.")
                log_proc.terminate()
                return True
            if time.time() - start_time > timeout:
                print("Timeout while waiting for server.")
                log_proc.terminate()
                return False
    except Exception as e:
        print("Error while checking logs:", str(e))
        log_proc.terminate()
        return False

def start_router_in_background(f_value):
    print(f"Starting router.py in background with -f {f_value}...")

    router_cmd = [
        "python3", "router.py",
        "-m", "resnet152",
        "-s", "4",
        "-f", str(f_value),
        "-p", "sa"
    ]

    process = subprocess.Popen(
        router_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return process

def wait_for_clients_by_log(container_id, f_value, timeout=900):
    print(f"Monitoring logs for final client readiness: func {f_value - 1}")

    target_line = f"Controller -- Receive signal 0 for func {f_value - 1}"
    start_time = time.time()

    log_proc = subprocess.Popen(
        ["docker", "logs", "-f", container_id],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    try:
        for line in log_proc.stdout:
            line = line.strip()
            if target_line in line:
                print(f"Detected: {line}")
                log_proc.terminate()
                return True
            if time.time() - start_time > timeout:
                print("Timeout while waiting for final client readiness.")
                log_proc.terminate()
                return False
    except Exception as e:
        print("Error reading container logs:", str(e))
        log_proc.terminate()
        return False

def run_sender_from_trace(f_value, d_value):
    print(f"Running sender_from_trace.py with arguments: {f_value} {d_value} 0")

    sender_cmd = [
        "python3", "sender_from_trace.py",
        str(f_value),
        str(d_value),
        "0"
    ]

    result = subprocess.run(sender_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("sender_from_trace.py execution failed:")
        print(result.stderr)
        return False

    print("sender_from_trace.py executed successfully.")
    print(result.stdout)
    return True

def stop_containers():
    print("Stopping standalone-client containers...")
    subprocess.run("docker ps -aq --filter ancestor=standalone-client | xargs -r docker stop", shell=True)
    print("Stopping standalone-server containers...")
    subprocess.run("docker ps -aq --filter ancestor=standalone-server | xargs -r docker stop", shell=True)
    print("Container cleanup complete.")

def analyze_router_log(f_value):
    print(f"Analyzing router.log with total_func = {f_value}")
    analyze_cmd = [
        "python3", "figure10_analyze_router_log.py",
        "router.log",
        "--total_func", str(f_value)
    ]
    result = subprocess.run(analyze_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print("Log analysis failed:")
        print(result.stderr)
    else:
        print("Log analysis completed:")
        print(result.stdout)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run standalone-server, router, sender_from_trace, and analyze")
    # parser.add_argument("-f", type=int, required=True, help="Value for the -f parameter (number of functions)")
    # parser.add_argument("-d", type=int, required=True, help="Second parameter to sender_from_trace.py (e.g. trace duration in minutes)")
    # args = parser.parse_args()

    parser = argparse.ArgumentParser(description="Run standalone-server, router, sender_from_trace, and analyze")
    parser.add_argument("-f", type=int, required=True, help="Value for the -f parameter (number of functions)")
    args = parser.parse_args()
    d_value = 5

    container_id = start_standalone_server()
    if container_id:
        if not wait_for_server_ready(container_id):
            print("Server did not start properly.")
        else:
            router_proc = start_router_in_background(args.f)
            all_ready = wait_for_clients_by_log(container_id, args.f)
            if not all_ready:
                print("Client loading not completed successfully.")
                router_proc.terminate()
            else:
                if run_sender_from_trace(args.f, d_value):
                # if run_sender_from_trace(args.f, args.d):
                    stop_containers()
                    analyze_router_log(args.f)
