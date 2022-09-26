import torch
from torch import nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight = nn.Parameter(torch.randn(10, 2))

    def forward(self, s):
        x = F.linear(s, self.weight)
        return x


model = MyModel()
x = torch.randn(1, 2)
output = model(x)
print(output)
output.mean().backward()
print(model.weight.grad)
model.zero_grad()

# Add another input feature
with torch.no_grad():
    model.weight = nn.Parameter(torch.cat((model.weight, torch.randn(10, 1)), 1))

x = torch.randn(1, 3)
output = model(x)
print(output)
output.mean().backward()
print(model.weight.grad)
model.zero_grad()