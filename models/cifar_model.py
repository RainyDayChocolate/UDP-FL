import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import NLLLoss, CrossEntropyLoss
import torchvision
from .base_model import BaseModel
import torch.nn.functional as F
from torch.nn.functional import softmax


class SimpleCifarCNN(nn.Module, BaseModel):
    def __init__(self, lr: float=0.01, max_norm: float = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Change the input channel size to 3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.max_norm = max_norm


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EfficientCifarCNN(BaseModel, nn.Module):
    def __init__(self, lr: float = 0.001, max_norm: float = 2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_norm = max_norm
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1).to(self.device)  # Change the input channel size to 3
        self.bn1 = nn.BatchNorm2d(32).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1).to(self.device)
        self.bn2 = nn.BatchNorm2d(64).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.fc1 = nn.Linear(64 * 8 * 8, 512).to(self.device)
        self.bn3 = nn.BatchNorm1d(512).to(self.device)
        self.fc2 = nn.Linear(512, 128).to(self.device)
        self.bn4 = nn.BatchNorm1d(128).to(self.device)
        self.fc3 = nn.Linear(128, 10).to(self.device)
        self.dropout = nn.Dropout(0.5).to(self.device)
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = CrossEntropyLoss(reduction='none').to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    

# Define a single residual block
# class ResidualBlock(nn.Module, BaseModel):
#     def __init__(self, in_channels, out_channels, stride=1):
#         super().__init__()
        
#         # First convolution layer
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
        
#         # Second convolution layer
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
        
#         # Shortcut connection to match dimensions
#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_channels != out_channels:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels)
#             )
        
#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out

# Define the ResNet model

class ResNet(nn.Module, BaseModel):
    def __init__(self, lr, max_norm: float = 2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm2d(64).to(self.device)
        self.relu = nn.ReLU(inplace=True).to(self.device)
        
        # Increase the number of blocks
        self.layer1 = self._make_layer(64, 128, 3, stride=1).to(self.device)
        self.layer2 = self._make_layer(128, 256, 4, stride=2).to(self.device)
        self.layer3 = self._make_layer(256, 512, 6, stride=2).to(self.device)
        
        # Use global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        
        self.fc = nn.Linear(512, 10).to(self.device)  # Change input dimension to 512
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(p=0.5).to(self.device)
        
        self.optimizer = Adam(self.parameters(), lr)
        self.loss_fn = CrossEntropyLoss(reduction='none').to(self.device)
        self.max_norm = max_norm
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride).to(self.device))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from .base_model import BaseModel  # Make sure to import your BaseModel

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # First 3x3 convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second 3x3 convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(BaseModel, nn.Module):
    def __init__(self, lr: float = 0.001, max_norm: float = 2):
        super(ResNet18, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_norm = max_norm

        # Initial 3x3 convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm2d(64).to(self.device)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1).to(self.device)
        self.layer2 = self._make_layer(64, 128, 2, stride=2).to(self.device)
        self.layer3 = self._make_layer(128, 256, 2, stride=2).to(self.device)
        self.layer4 = self._make_layer(256, 512, 2, stride=2).to(self.device)
        
        # Global average pooling and fully connected layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(self.device)
        self.fc = nn.Linear(512, 100).to(self.device)  # 100 classes for CIFAR-100
        
        # Optimizer and loss function
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = CrossEntropyLoss(reduction='none').to(self.device)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(in_channels, out_channels, stride).to(self.device))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class resnet18buildin(BaseModel, torch.nn.Module):
    def __init__(self, lr: float = 0.001, max_norm: float = 2, num_classes: int = 10):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        n_ftrs = self.backbone.fc.in_features
        self.backbone.fc = torch.nn.Linear(n_ftrs, num_classes)
        
        # Move the model to CUDA
        self.to('cuda')
        
    def forward(self, x):
        # Ensure the input is on CUDA
        x = x.to('cuda')
        
        logits = self.backbone(x)
        return logits, softmax(logits, dim=-1)