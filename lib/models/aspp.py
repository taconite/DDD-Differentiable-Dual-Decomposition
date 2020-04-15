import torch
import torch.nn as nn
from torch.nn import functional as F

class ASPP(nn.Module):

    def __init__(self, inplanes, depth, conv=nn.Conv2d, norm=nn.BatchNorm2d, momentum=0.0003, mult=1):
        super(ASPP, self).__init__()
        self._inplanes = inplanes
        self._depth = depth

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.aspp1 = conv(inplanes, depth, kernel_size=1, stride=1, bias=False)
        self.aspp2 = conv(inplanes, depth, kernel_size=3, stride=1,
                               dilation=int(6*mult), padding=int(6*mult),
                               bias=False)
        self.aspp3 = conv(inplanes, depth, kernel_size=3, stride=1,
                               dilation=int(12*mult), padding=int(12*mult),
                               bias=False)
        self.aspp4 = conv(inplanes, depth, kernel_size=3, stride=1,
                               dilation=int(18*mult), padding=int(18*mult),
                               bias=False)
        self.aspp5 = conv(inplanes, depth, kernel_size=1, stride=1, bias=False)
        self.aspp1_bn = norm(depth, momentum=momentum)
        self.aspp2_bn = norm(depth, momentum=momentum)
        self.aspp3_bn = norm(depth, momentum=momentum)
        self.aspp4_bn = norm(depth, momentum=momentum)
        self.aspp5_bn = norm(depth, momentum=momentum)
        self.conv2 = conv(depth * 5, depth, kernel_size=1, stride=1,
                               bias=False)
        self.bn2 = norm(depth, momentum=momentum)

    def forward(self, x):
        x1 = self.aspp1(x)
        x1 = self.aspp1_bn(x1)
        x1 = self.relu(x1)
        x2 = self.aspp2(x)
        x2 = self.aspp2_bn(x2)
        x2 = self.relu(x2)
        x3 = self.aspp3(x)
        x3 = self.aspp3_bn(x3)
        x3 = self.relu(x3)
        x4 = self.aspp4(x)
        x4 = self.aspp4_bn(x4)
        x4 = self.relu(x4)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = self.aspp5_bn(x5)
        x5 = self.relu(x5)
        x5 = F.interpolate(x5, (x.shape[2], x.shape[3]), mode='bilinear',
                           align_corners=False)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
