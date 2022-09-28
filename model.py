from torchsummary import summary
from torchvision.models import resnet18, ResNet50_Weights
import torch
from torch import nn


class my_resnet(nn.Module):
    def __init__(self):
        super(my_resnet, self).__init__()
        self.conv = nn.Conv2d(1,3,(1,1),1,0)
        self.res = resnet18()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000,2)
        )
    def forward(self,x):

        x = self.res(x)
        x = self.fc(x)
        return x


