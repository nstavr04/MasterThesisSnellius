### Import necessary libraries and dependencies
import torch
import torch.nn as nn
import math

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        ### Compute channels for both primary and secondary based on ratio (s)
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

		### Primary standard convolution + BN + ReLU
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

		### Secondary depthwise convolution + BN + ReLU
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        ### Stack
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]
