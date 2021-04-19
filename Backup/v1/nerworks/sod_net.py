import torch
import torch.nn as nn

from .sod_net_part import ConvBlock, DownBlock, UpBlock, BPR

class SODNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64,
                 depth=5, width=2, down_method="conv", up_method="deconv"):
        super().__init__()

        self.depth = depth
        width = width if width > 1 else 1 
        channel_list = [base_channels * 2 ** i for i in range(depth)]

        self.down_conv_list = nn.ModuleList([
                ConvBlock(in_channels, channel_list[0], nums_channel=width)])
        self.down_list = nn.ModuleList()
        self.up_conv_list = nn.ModuleList()
        self.up_list = nn.ModuleList()
        
        for i in range(depth-1):
            self.down_conv_list.append(
                ConvBlock(channel_list[i], channel_list[i+1], nums_channel=width))
            self.down_list.append(
                DownBlock(down_method, channel_list[i], channel_list[i]))
            
            self.up_conv_list.append(
                ConvBlock(channel_list[depth - i - 1], channel_list[depth - i - 2], nums_channel=width))
            self.up_list.append(
                UpBlock(up_method, channel_list[depth - i - 1], channel_list[depth - i - 2]))
        
        self.conv = nn.Conv2d(channel_list[0], out_channels, kernel_size=1)
        self.bpr = BPR(channel_list[0], out_channels)

    def forward(self, x):
        # down
        x_down_list = [self.down_conv_list[0](x)]
        for i in range(self.depth - 1):
            temp = self.down_list[i](x_down_list[i])
            temp = self.down_conv_list[i + 1](temp)
            x_down_list.append(temp)
        
        # up
        x_up = x_down_list[-1]
        for i in range(self.depth - 1):
            temp = self.up_list[i](x_up)
            temp = torch.cat([x_down_list[self.depth - i - 2], temp], dim=1)
            x_up = self.up_conv_list[i](temp)
        
        y = self.conv(x_up)
        new_y = self.bpr(x_down_list[0], y)
        return y, new_y