import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, 
                                        kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 
                                        kernel_size=1, padding=0)
        
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, nums_channel=1):
        super().__init__()

        Conv2d = DepthwiseSeparableConv

        self.in_conv = nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.out_convs = nn.ModuleList([])
        for i in range(nums_channel-1):
            self.out_convs.append(nn.Sequential(
                Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

    def forward(self, x):
        x = self.in_conv(x)
        for module in self.out_convs:
            x = module(x)   
        return x


class DownBlock(nn.Module):
    def __init__(self, method="pooling", in_channel=None, out_channel=None):
        super().__init__()
        assert method in ["pooling", "conv"]

        if method == "pooling":
            self.down = nn.MaxPool2d(2)
        elif method == "conv":
            assert in_channel is not None and out_channel is not None, "channel can't be None."
            self.down = nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.down(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, method="sample", in_channel=None, out_channel=None):
        super().__init__()
        assert method in ["sample", "deconv"]
        if method == "sample":
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif method == "deconv":
            assert in_channel is not None and out_channel is not None, "channel can't be None."
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x

class BPR(nn.Module):
    '''
       Boundary preserved reﬁnement 
    '''
    def __init__(self, x_channel, y_channel):
        super().__init__()

        Conv2d = DepthwiseSeparableConv

        self.conv_1 = nn.Sequential(
            Conv2d(x_channel, x_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(x_channel),
            nn.ReLU(inplace=True),
        )
        self.conv_2 = nn.Sequential(
            Conv2d(x_channel + y_channel, x_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(x_channel),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Conv2d(x_channel, y_channel, kernel_size=1)
    
    def forward(self, x, y):
        x = self.conv_1(x)
        new_y = torch.cat((x, y), dim=1)
        new_y = self.conv_2(new_y)
        new_y = self.out_conv(new_y)
        return new_y