from torch import nn
from torch.optim import SGD
from torch.nn import NLLLoss
from .base_model import BaseModel

# Demo model for MNIST
class MnistFullConnectModel(BaseModel, nn.Module):
    def __init__(self):
        optimizer = SGD
        loss_fn = NLLLoss()
        optimizer_kwargs = {"lr": 0.01}
        super(BaseModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)
        self.optimizer = optimizer(self.parameters(), **optimizer_kwargs)
        self.loss_fn = loss_fn

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
