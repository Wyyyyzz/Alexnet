
import torch.nn as nn
import torch
import torch.nn.functional as ff

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.c1=nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4,padding=2),
        self.ReLU=nn.ReLU(),
        self.s1=nn.MaxPool2d(kernel_size=3, stride=2),
        self.c2=nn.Conv2d(in_channels=48,out_channels=128,kernel_size=5,stride=1,padding=2),#第一层卷积，激活，池化层
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2),
        self.c3=nn.Conv2d(in_channels=128,out_channels=192,kernel_size=3,stride=1,padding=2),
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=2),
        self.c5 = nn.Conv2d(in_channels=192,out_channels=128, kernel_size=3, stride=1, padding=2),
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2),
        self.flatten = nn.Flatten(),
        self.f6=nn.Linear(4608,2048),
        self.f7=nn.Linear(2048,2048),
        self.f8=nn.Linear(2048,1000),
        self.f9=nn.Linear(1000,2),

    def forward(self,x):
        x = self.ReLU(self.c1(x)),
        x = self.s1(x),
        x = self.ReLU(self.c2(x)),
        x = self.s2(x),
        x = self.ReLU(self.c3(x)),
        x = self.ReLU(self.c4(x)),
        x = self.ReLU(self.c5(x)),
        x = self.s5(x),
        x = self.flatten(x),
        x = self.f6(x),
        x = ff.dropout(x, p=0.5),#为了防止过拟合，神经网络随机一半失效
        x = self.f7(x),
        x = ff.dropout(x, p=0.5),
        x = self.f8(x),
        x = ff.dropout(x, p=0.5),
        return x

if __import__=="__main__":
    x = torch.rand([1,3,224,224])
    model = AlexNet()
    y = model


