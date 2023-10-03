import torch
from torch import nn
from torch.optim import SGD, Adam
from torch.nn import NLLLoss, CrossEntropyLoss
import torchvision
from .base_model import BaseModel
import torch.nn.functional as F
from torch.nn.functional import softmax


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
class ResidualBlock(nn.Module, BaseModel):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
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

#Define the ResNet model

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
        
        self.optimizer = Adam(self.parameters(), lr, weight_decay=1e-4)
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

class ResidualBlock18(nn.Module, BaseModel):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
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

class ResNet18(nn.Module, BaseModel):
    def __init__(self, lr: float = 0.001, max_norm: float = 2):
        super().__init__()
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
        self.fc = nn.Linear(512, 10).to(self.device)  # 100 classes for CIFAR-100
        
        # Optimizer and loss function
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = CrossEntropyLoss(reduction='none').to(self.device)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock18(in_channels, out_channels, stride))
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


class BasicBlock(nn.Module, BaseModel):

    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x))).to(self.device)
        out = self.bn2(self.conv2(out)).to(self.device)
        out += self.shortcut(x).to(self.device)
        out = F.relu(out).to(self.device)
        return out


class Bottleneck(nn.Module, BaseModel):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))).to(self.device)
        out = F.relu(self.bn2(self.conv2(out))).to(self.device)
        out = self.bn3(self.conv3(out)).to(self.device)
        out += self.shortcut(x).to(self.device)
        out = F.relu(out).to(self.device)
        return out


class ResNet101(nn.Module, BaseModel):
    def __init__(self, block=Bottleneck, num_blocks=[3, 4, 23, 3], lr: float = 0.001, max_norm: float = 2, num_classes: int = 10):
        super().__init__()
        self.num_classes = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.lr = lr
        self.max_norm = max_norm
        self.in_planes = 64
        self.relu_1 = nn.ReLU().to(self.device)
        # Initialize layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False).to(self.device)
        self.bn1 = nn.BatchNorm2d(64).to(self.device)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1).to(self.device)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2).to(self.device)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2).to(self.device)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2).to(self.device)
        self.linear = nn.Linear(512 * block.expansion, num_classes).to(self.device)
        
        # Optimizer and loss function
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride).to(self.device))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.device)
        out = self.relu_1(self.bn1(self.conv1(x))).to(self.device)
        out = self.layer1(out).to(self.device)
        out = self.layer2(out).to(self.device)
        out = self.layer3(out).to(self.device)
        out = self.layer4(out).to(self.device)
        out = F.avg_pool2d(out, 4).to(self.device)
        out = out.view(out.size(0), -1).to(self.device)
        out = self.linear(out).to(self.device)
        return out