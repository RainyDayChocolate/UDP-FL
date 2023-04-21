from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.nn import NLLLoss, CrossEntropyLoss
from .base_model import BaseModel
from torch.optim import SGD, Adam
from torch.nn import NLLLoss, CrossEntropyLoss

class MedicalCNN(BaseModel,nn.Module):
    def __init__(self, lr, max_norm: float = 2):
        super(MedicalCNN, self).__init__()
        self.conv1=nn.Conv2d(3,6,3,1)
        self.conv2=nn.Conv2d(6,16,3,1)
        self.fc1 = nn.Linear(16 * 14 * 14, 120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,20)
        self.fc4=nn.Linear(20,6)
        self.optimizer = Adam(self.parameters(), lr)
        self.loss_fn = CrossEntropyLoss(reduction='none')
        self.max_norm = max_norm
        
    def forward(self,X):
        X=F.relu(self.conv1(X))
        X=F.max_pool2d(X,2,2)
        X=F.relu(self.conv2(X))
        X=F.max_pool2d(X,2,2)
        #print("Shape after last pooling layer:", X.shape)
        X=X.view(-1,16*14*14)
        X=F.relu(self.fc1(X))
        X=F.relu(self.fc2(X))
        X=F.relu(self.fc3(X))
        X=self.fc4(X)
        return F.log_softmax(X,dim=1)
