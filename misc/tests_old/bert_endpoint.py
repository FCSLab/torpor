import time
import sys
import json
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
import os
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', stream=sys.stdout, level=logging.INFO)

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained(
    'bert-large-uncased-whole-word-masking-finetuned-squad').cuda()
model.eval()
model = model.cuda()
torch.cuda.synchronize()

def inf():
    x = torch.cat((torch.ones((1, 512), dtype=int).view(-1), torch.ones((1, 512), dtype=int).view(-1))).view(2, -1, 512).cuda()
    start_t = time.time()
    with torch.no_grad():
        y = model(x[0], token_type_ids=x[1])
        output = y[0][0].sum().to('cpu')
    end_t = time.time()
    del x

    print(f'start {start_t}, end {end_t}, elasped {end_t - start_t}')
    msg = "output: {}, elasped: {}".format(output, end_t - start_t)

    return msg

class MyRequest(BaseHTTPRequestHandler):

    def inference(self):
        return inf()

    def reply(self, msg):
        data = {"result": msg}
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def core(self):
        cur_server = self.headers.get('cur_server')
        if cur_server is not None:
            os.environ['CUR_SERVER_ID'] = cur_server
        msg = self.inference()
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
    # inf()
    server = HTTPServer(host, MyRequest)
    logging.info("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
