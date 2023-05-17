import torch.nn.functional as F
import torch.nn as nn
from lib.module.layer.deform_conv_v2 import DeformConv2d
import torch

class DeformLeNet(nn.Module):
    def __init__(self):
        super(DeformLeNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv1 = DeformConv2d(3, 6, 3, padding=1, modulation=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = DeformConv2d(6, 16, 3, padding=1, modulation=True)
        self.fc1 = nn.Linear(16*8*8, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*8*8)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__':
    net = DeformLeNet().cuda()
    in1 = torch.randn((4,3,32,32)).cuda()
    out1 = net(in1)
    print(out1.size())