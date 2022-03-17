import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class RGFSConv(nn.Module):
    def __init__( self, input_channel, output_channel, ratio, kernel_size=3, padding=0, stride=1, bias=True, dilation=1):
        super(RGFSConv, self).__init__()
        
        self.output_channel = output_channel
        self.conv1 = nn.Conv2d(input_channel, int(output_channel*ratio), kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, dilation=dilation)
    def forward(self, x, mask):
        inter_feature = self.conv1(x)
        inter_feature = inter_feature.permute(0, 2, 3, 1)
        mask = F.interpolate(torch.unsqueeze(mask, 1), scale_factor=0.25, mode='nearest')
        mask = torch.squeeze(mask, 1)

        y = torch.zeros((inter_feature.size()[:3] + (self.output_channel,))).permute(0, 3, 1, 2).cuda()
        for i in range(10):
            index = torch.zeros((x.size()[0], x.size()[2], x.size()[3], 1)).cuda()
            index[mask==i] = 1
            temp = torch.mul(inter_feature, index)
            sum_ = temp.sum(dim=1).sum(dim=1)
            _, indices = torch.sort(sum_, descending=True)
            e, _ = indices[:, :self.output_channel].sort()
            for j in range(inter_feature.size()[0]):
                y[j] += temp.permute(0, 3, 1, 2)[j, e[j], :, :]
        return y