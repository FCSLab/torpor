import time
print(f'Start: {time.time()}')
import torch
import numpy as np
import torchvision.models as models
print(f'Module: {time.time()}')

torch.tensor([1]).cuda()
print(f'cuda init: {time.time()}')

model = models.resnet50(pretrained=True)
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
    time.sleep(0.2)
    elasped.append(inf())

print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')