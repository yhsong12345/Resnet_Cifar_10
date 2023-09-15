import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *



def SelectModel(m):
    
    if m == 'Resnet20':
        return ResNet(BasicBlock, [3, 3, 3])
    elif m == 'Resnet32':
        return ResNet(BasicBlock, [5, 5, 5])
    elif m == 'Resnet44':
        return ResNet(BasicBlock, [7, 7, 7])
    elif m == 'Resnet56':
        return ResNet(BasicBlock, [9, 9, 9])
    elif m == 'Resnet110':
        return ResNet(Bottleneck, [18, 18, 18])
    elif m == 'Resnet164':
        return ResNet(Bottleneck, [27, 27, 27])
    elif m == 'Plain20':
        return ResNet(PlainBasicBlock, [3, 3, 3])
    elif m == 'Plain32':
        return ResNet(PlainBasicBlock, [5, 5, 5])
    



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        print(strides)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes*block.expansion
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.layer1(o)
        o = self.layer2(o)
        o = self.layer3(o)
        o = F.avg_pool2d(o, o.size()[3])
        o = o.view(o.size(0), -1)
        o = self.linear(o)

        return o

    
