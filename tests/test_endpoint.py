import time
import os
import sys
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', stream=sys.stdout, level=logging.INFO)

class MyRequest(BaseHTTPRequestHandler):
    def reply(self, msg):
        data = {"result": msg}
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def core(self):
        msg = f'received at {time.time()}'
        self.reply(msg)

    def do_GET(self):
        self.core()

    def do_POST(self):
        self.core()

if __name__ == "__main__":
    port = 9000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    print(f'using port {port}')
    host = ("0.0.0.0", port)
    server = HTTPServer(host, MyRequest)
    logging.info("Starting server, listen at: %s:%s" % host)
    server.serve_forever()