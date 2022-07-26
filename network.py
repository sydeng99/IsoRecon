import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from torch.autograd import Variable
from torch.nn import init
from math import sqrt



class GResidualBlock(nn.Module):
    def __init__(self,in_c):
        super(GResidualBlock,self).__init__()

        conv_block=[  nn.Conv2d(in_c,in_c,3,1,1),
                      nn.ReLU(inplace=True),
                      nn.Conv2d(in_c,in_c,3,1,1)]

        self.conv_block=nn.Sequential(*conv_block)

    def forward(self, x):
        return x+self.conv_block(x)




class Generator1(nn.Module):
    def __init__(self,in_c,out_c,residual_blocks=9):
        super(Generator1,self).__init__()

        self.Ave=nn.AvgPool2d((10,1),stride=(10,1),padding=(0,0))

        model=[  self.Ave,
                 nn.Conv2d(in_c, 64,3,1,1),
                 nn.ReLU(inplace=True)
        ]

        for _ in range(residual_blocks):
            model+=[GResidualBlock(64)]

        model += [nn.Conv2d(64,out_c,kernel_size=3,stride=1,padding=1)]

        self.model=nn.Sequential(*model)

    def forward(self, x):
        return self.Ave(x)+self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)




class Generator2(nn.Module):
    def __init__(self,in_c,out_c,residual_blocks=9):
        super(Generator2,self).__init__()

        model = [nn.Upsample(scale_factor=(10, 1), mode='bilinear')]

        model+=[  nn.Conv2d(in_c,64,3,1,1),
                 nn.ReLU(inplace=True)]

        for _ in range(residual_blocks):
            model+=[GResidualBlock(64)]

        model+=[nn.Conv2d(64,out_c,3,1,1)]

        self.model = nn.Sequential(*model)
        self.upsample = nn.Upsample(scale_factor=(10, 1), mode='bilinear')
        self._initialize_weights()

    def forward(self, x):
        return self.upsample(x) + self.model(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)


