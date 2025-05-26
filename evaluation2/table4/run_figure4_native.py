import warnings
warnings.filterwarnings("ignore")
import time
import torch
import numpy as np
import torchvision.models as models
from transformers import BertTokenizer, BertForQuestionAnswering

model_names = [
    "densenet169",
    "densenet201",
    "inception_v3",
    "efficientnet_b0",
    "resnet50",
    "resnet101",
    "resnet152"
]

torch.tensor([1]).cuda()

for model_name in model_names:
    try:
        model_fn = getattr(models, model_name)
        model = model_fn(pretrained=True)
        model.eval()
        model = model.cuda()
        torch.cuda.synchronize()

        def inf():
            # Note: inception_v3 needs input size 299x299
            size = 299 if "inception" in model_name else 224
            x = torch.ones((1, 3, size, size)).cuda()
            start_t = time.time()
            with torch.no_grad():
                y = model(x)
                output = y.sum().to('cpu')
            end_t = time.time()
            del x
            return end_t - start_t

        inf()  # warm-up

        elapsed = []
        for _ in range(10):
            time.sleep(0.2)
            elapsed.append(inf())

        print(f"{model_name}: Latency avg {np.average(elapsed):.6f}, std {np.std(elapsed):.6f}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Error with model {model_name}: {e}")

try:
    model = BertForQuestionAnswering.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad"
    ).cuda().eval()

    def bert_inf(batch_size=1):
        x = torch.cat(
            (
                torch.ones((batch_size, 512), dtype=torch.long).view(-1),
                torch.ones((batch_size, 512), dtype=torch.long).view(-1),
            )
        ).view(2, -1, 512).cuda()

        input_ids = x[0]
        token_type_ids = x[1]

        start_t = time.time()
        with torch.no_grad():
            y = model(input_ids=input_ids, token_type_ids=token_type_ids)
            _ = y.start_logits.sum() + y.end_logits.sum()
        end_t = time.time()
        del x
        return end_t - start_t

    bert_inf()  # warm-up

    elapsed = []
    for _ in range(10):
        time.sleep(0.2)
        elapsed.append(bert_inf(batch_size=1))

    print(f"bert-qa: Latency avg {np.average(elapsed):.6f}, std {np.std(elapsed):.6f}")

    del model
    torch.cuda.empty_cache()

except Exception as e:
    print(f"Error with BERT-QA: {e}")