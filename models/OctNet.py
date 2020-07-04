import torch.nn.functional as F
from models.layer.OctConv import *

class OctNet(nn.Module):
    def __init__(self):
        """
        OctNet是将LeNet中的卷积替换为OctConv实现的net
        """
        super(OctNet, self).__init__()
        self.convhead = FirstOctaveCBR(kernel_size=(3, 3), in_channels=3, out_channels=64,stride=1,alpha=0.5)
        self.convbody1 = OctaveCBR(kernel_size=(3, 3), in_channels=64, out_channels=128, bias=False, stride=2,alpha=0.5)
        self.convbody2 = OctaveCBR(kernel_size=(3, 3), in_channels=128, out_channels=128, bias=False, stride=2,alpha=0.5)
        self.convtail = LastOctaveCBR(kernel_size=(3, 3), in_channels=128, out_channels=128, bias=False, stride=2,alpha=0.5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # x:  3,32,32
        x = self.convhead(x)   # 32,32,32 + 32,16,16
        x = self.convbody1(x)  # 64,16,16 + 64,8,8
        x = self.convbody2(x)  # 64,8,8 + 64,4,4
        x = self.convtail(x)   # 128,4,4
        x = x.view(-1, 128*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    model = OctNet()
    inp= torch.randn(10,3,32,32)
    oup = model(inp)
    print(oup.size())