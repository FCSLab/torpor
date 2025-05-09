import time
import os
import sys
import json
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%Y-%m-%d,%H:%M:%S', stream=sys.stdout, level=logging.INFO)

import torchvision.models as models

model = models.resnet50(pretrained=True)
model = models.resnet101(pretrained=True)
model = models.resnet152(pretrained=True)
model = models.densenet169(pretrained=True)
model = models.densenet201(pretrained=True)
model = models.inception_v3(pretrained=True)
model = models.efficientnet_b0(pretrained=True)

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
