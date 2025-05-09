import os

os.environ["CUDNN_LOGINFO_DBG"] = '1'
os.environ["CUBLAS_LOGINFO_DBG"] = '1'
os.environ["CUDNN_LOGDEST_DBG"] = 'cudnn.log'
os.environ["CUBLAS_LOGDEST_DBG"] = 'cublas.log'

import torch
from torch.nn import Conv1d

a = torch.ones((10, 1, 1)).cuda()
conv = Conv1d(1, out_channels=1, kernel_size=1)
conv.to(device='cuda')
b = conv(a)
print(b.to('cpu'))