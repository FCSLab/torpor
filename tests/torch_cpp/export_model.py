import torch
import torchvision

# get model
model = torchvision.models.vgg16()

# generate random example
example = torch.rand(1, 3, 224, 224)

# use torch.jit.trace to generate torch.jit.ScriptModule
traced_script_module = torch.jit.trace(model, example)

output = traced_script_module(torch.ones(1, 3, 224, 224))
print(output[0, :5])

traced_script_module.save("model.pt")