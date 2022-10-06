from torchsummary import summary
from torchvision.models import resnet34,resnet18,resnet50
import torch.nn.functional as F
import torch
from torch import nn


class resnet_34(nn.Module):
    def __init__(self):
        super(resnet_34, self).__init__()
        # self.conv = nn.Conv2d(1,3,(1,1),1,0)
        self.res = resnet34()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000,2),
            nn.Softmax(dim=1),
        )
    def forward(self,x):

        x = self.res(x)
        x = self.fc(x)
        return x

class resnet_50(nn.Module):
    def __init__(self):
        super(resnet_50, self).__init__()
        # self.conv = nn.Conv2d(1,3,(1,1),1,0)
        self.res = resnet50()
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000,2),
            nn.Softmax(dim=1),
        )
    def forward(self,x):

        x = self.res(x)
        x = self.fc(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, activation=True):
        super(Conv, self).__init__()
        padding = kernel_size // 2 if padding is None else padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, groups=groups, bias=True)
        self.act = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

class VGG_19(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG_19, self).__init__()
        self.stages = nn.Sequential(*[
            self._make_stage(3, 64, num_blocks=2, max_pooling=True),
            self._make_stage(64, 128, num_blocks=2, max_pooling=True),
            self._make_stage(128, 256, num_blocks=4, max_pooling=True),
            self._make_stage(256, 512, num_blocks=4, max_pooling=True),
            self._make_stage(512, 512, num_blocks=4, max_pooling=True)
        ])
        self.head = nn.Sequential(*[
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        ])

    @staticmethod
    def _make_stage(in_channels, out_channels, num_blocks, max_pooling):
        layers = [Conv(in_channels, out_channels, kernel_size=3, stride=1)]
        for _ in range(1, num_blocks):
            layers.append(Conv(out_channels, out_channels, kernel_size=3, stride=1))
        if max_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.head(self.stages(x))




if __name__ == '__main__':
    model = VGG_19()
    #model = resnet_34()
    img = torch.randn(2, 3, 224, 224)
    print(model(img))


