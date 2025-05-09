import time
print(f'Start: {time.time()}')

import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from transformers import BertForQuestionAnswering
print(f'Module: {time.time()}')

torch.tensor([1]).cuda()
print(f'cuda init: {time.time()}')

model = model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model.eval()
print(f'To host mem: {time.time()}')

model = model.cuda()
torch.cuda.synchronize()
print(f'To GPU mem: {time.time()}')

# x = torch.cat((torch.randint(5000, size=[1, 512]).view(-1), torch.randint(low=0, high=2, size=[1, 512]).view(-1))).view(2, -1, 512).cuda()
x = torch.cat((torch.ones((1, 512), dtype=int).view(-1), torch.ones((1, 512), dtype=int).view(-1))).view(2, -1, 512).cuda()

def inf():
    start_t = time.time()
    with torch.no_grad():
        y = model(x[0], token_type_ids=x[1])
        output = y[0][0].sum().to('cpu')
    end_t = time.time()
    # torch.cuda.empty_cache()
    print(f'output {output}, elasped {end_t - start_t}')
    return end_t - start_t


inf()
print(f'First inf: {time.time()}')

elasped = []
for i in range(3):
    # print('sleep 1s')
    time.sleep(1)
    elasped.append(inf())
print(f'Latency avg {np.average(elasped)}, std {np.std(elasped)}')