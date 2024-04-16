import time
print(f'Start: {time.time()}')
import torch
import numpy as np
import os
import sys
import torchvision.models as models
print(f'Module: {time.time()}')

torch.tensor([1]).cuda()
print(f'cuda init: {time.time()}')

model_name = 'resnet152'
if (len(sys.argv) > 1):
    model_name = sys.argv[1]

if model_name == 'resnet50':
    model = models.resnet50(pretrained=True)
elif model_name == 'resnet101':
    model = models.resnet101(pretrained=True)
elif model_name == 'vgg':
    model = models.vgg16(pretrained=True)
elif model_name == 'densenet':
    model = models.densenet169(pretrained=True)
elif model_name == 'squeeze':
    model = models.squeezenet1_1(pretrained=True)
elif model_name == 'resnet152':
    model = models.resnet152(pretrained=True)
elif model_name == 'mobilenet':
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
elif model_name == 'inception':
    model = models.inception_v3(pretrained=True)
else:
    print(f'Unknown model {model_name}')
    exit(1)

model.eval()
print(f'To host mem: {time.time()}')

model = model.cuda()
torch.cuda.synchronize()
print(f'To GPU mem: {time.time()}')

def inf():
    x = torch.ones((1, 3, 224, 224)).cuda()
    start_t = time.time()
    with torch.no_grad():
        y = model(x)
        output = y.sum().to('cpu')
    end_t = time.time()
    del x
    print(f'output {output}, elasped {end_t - start_t}')
    return end_t - start_t

inf()
print(f'First inf: {time.time()}')

elasped = []
for i in range(10):
    # print('sleep 1s')
    time.sleep(0.2)
    elasped.append(inf())

print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')
