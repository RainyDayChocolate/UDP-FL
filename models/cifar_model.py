from torch import nn
from torch.optim import SGD, Adam
from torch.nn import NLLLoss, CrossEntropyLoss
from .base_model import BaseModel
import torch.nn.functional as F

class SimpleCifarCNN(BaseModel, nn.Module):
    def __init__(self,learningrate: float=0.01):
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
    

class EfficientCifarCNN(BaseModel, nn.Module):
    def __init__(self, learning_rate: float = 0.001):
        super(EfficientCifarCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Change the input channel size to 3
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
        self.optimizer = Adam(self.parameters(), lr = learning_rate)
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x