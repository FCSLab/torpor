import time
print(f'Start: {time.time()}')
import torch
import numpy as np
import torchvision.models as models
print(f'Module: {time.time()}')

# 初始化 CUDA
torch.tensor([1]).cuda()
print(f'cuda init: {time.time()}')

# 加载 EfficientNet 模型，这里以 EfficientNet-B0 为例
model = models.efficientnet_b0(pretrained=True)
model.eval()
print(f'To host mem: {time.time()}')

# 将模型移动到 GPU
model = model.cuda()
torch.cuda.synchronize()
print(f'To GPU mem: {time.time()}')

def inf():
    # EfficientNet 的输入尺寸为 (1, 3, 224, 224)
    x = torch.ones((1, 3, 224, 224)).cuda()
    start_t = time.time()
    with torch.no_grad():
        y = model(x)
        output = y.sum().to('cpu')
    end_t = time.time()
    del x
    print(f'output {output}, elasped {end_t - start_t}')
    return end_t - start_t

# 进行第一次推理
inf()
print(f'First inf: {time.time()}')

elasped = []
for i in range(10):
    time.sleep(0.2)
    elasped.append(inf())

# 计算平均延迟和标准差
print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')