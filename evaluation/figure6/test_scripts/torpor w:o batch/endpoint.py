import time
import os
import sys
import json
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', stream=sys.stdout, level=logging.INFO)

import torchvision.models as models

model_name = 'resnet152'
if 'model_name' in os.environ:
    model_name = os.environ['model_name']

if model_name == 'resnet50':
    model = models.resnet50(pretrained=True)
elif model_name == 'resnet101':
    model = models.resnet101(pretrained=True)
elif model_name == 'vgg':
    model = models.vgg16(pretrained=True)
elif model_name == 'densenet169':
    model = models.densenet169(pretrained=True)
elif model_name == 'densenet201':
    model = models.densenet201(pretrained=True)
elif model_name == 'squeeze':
    model = models.squeezenet1_1(pretrained=True)
elif model_name == 'resnet152':
    model = models.resnet152(pretrained=True)
elif model_name == 'mobilenet':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
elif model_name == 'inception':
    model = models.inception_v3(pretrained=True)
elif model_name == 'efficientnet':
    model = models.efficientnet_b0(pretrained=True)
elif model_name == 'bertqa':
    from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
    from transformers import BertForQuestionAnswering

    model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
else:
    print(f'Unknown model {model_name}')
    exit(1)

model.eval()
model = model.cuda()
# x = torch.ones((1, 3, 224, 224)).cuda()
logging.info(f"Model loaded: {model_name}") # print model_name
torch.cuda.synchronize()

class MyRequest(BaseHTTPRequestHandler):
    def bert_inf(self, batch_size=1):
        x = torch.cat((torch.ones((batch_size, 512), dtype=int).view(-1), torch.ones((batch_size, 512), dtype=int).view(-1))).view(2, -1, 512).cuda()
        start_t = time.time()
        with torch.no_grad():
            y = model(x[0], token_type_ids=x[1])
            output = y[0][0].sum().to('cpu')
        end_t = time.time()
        del x

        return output, start_t, end_t

    def cv_inf(self, batch_size=1):
        start_t = time.time()
        x = torch.ones((batch_size, 3, 224, 224)).cuda(non_blocking=True)
        # x = torch.ones((1, 3, 224, 224)).cuda()
        # mid_t = time.time()
        with torch.no_grad():
            y = model(x)
            # torch.cuda.default_stream().synchronize()
            # exec_t = time.time()
            output = y.sum().to('cpu')
        end_t = time.time()
        del x

        return output, start_t, end_t

    def inference(self, batch_size=1):
        if model_name == 'bertqa':
            output, start_t, end_t = self.bert_inf(batch_size)
        else:
            output, start_t, end_t = self.cv_inf(batch_size)
        logging.info(f'start {start_t}, end {end_t}, elasped {end_t - start_t}')
        # msg = "output: {}, start_t: {}, input: {}, exec: {}, elasped: {}".format(output, start_t, mid_t - start_t, exec_t - mid_t, end_t - start_t)
        msg = "output: {}, start_t: {}, elasped: {}".format(output, start_t, end_t - start_t)

        return msg

    def reply(self, msg):
        data = {"result": msg}
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def core(self):
        logging.info(f'receive req')
        cur_server = self.headers.get('cur_server')
        if cur_server is not None:
            os.environ['CUR_SERVER_ID'] = cur_server

        bs = 1
        cur_batch_size = self.headers.get('batch_size')
        if cur_batch_size is not None:
            bs = int(cur_batch_size)
        msg = self.inference(bs)
        self.reply(msg)

    def do_GET(self):
        self.core()

    def do_POST(self):
        self.core()

if __name__ == "__main__":
    port = 9000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    print(f'Model {model_name} on port {port}')
    host = ("0.0.0.0", port)
    server = HTTPServer(host, MyRequest)
    logging.info("Starting server, listen at: %s:%s" % host)
    server.serve_forever()

