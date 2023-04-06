from torch import nn
from torch.optim import SGD, Adam
from torch.nn import NLLLoss, CrossEntropyLoss
from .base_model import BaseModel
import torch.nn.functional as F

class SimpleCifarCNN(BaseModel, nn.Module):
    def __init__(self,learningrate: float=0.1):
        super(SimpleCifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Change the input channel size to 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = Adam(self.parameters(), lr=learningrate)
        self.loss_fn = CrossEntropyLoss()
        

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x