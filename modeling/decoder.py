import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.RGFSConv import RGFSConv


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, ratio, input_heads=1):
        super(Decoder, self).__init__()
        
        if backbone == 'resnet' or backbone == 'drn' :
            low_level_inplanes = 256
            last_conv_input = 304
        elif backbone == 'resnet_adv'or backbone=='resnet_condconv':
            low_level_inplanes = 256*input_heads
            last_conv_input = 256*input_heads + 48
        elif backbone == 'xception':
            low_level_inplanes = 128
            last_conv_input = 304
        elif backbone == 'xception_adv':
            low_level_inplanes = 128*input_heads
            last_conv_input = 256*input_heads + 48
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
            last_conv_input = 304
        elif backbone == 'plus':
            low_level_inplanes = 256
            last_conv_input = 304
        else:
            raise NotImplementedError
        
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.condconv1 = RGFSConv(last_conv_input, 256, ratio, kernel_size=3, stride=1, padding=1, bias=False)
        self.last_conv = nn.Sequential(
                                    BatchNorm(256),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                    BatchNorm(256),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat, mask):

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.condconv1(x, mask)
        x = self.last_conv(x)


        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, ratio, input_heads=1):
    return Decoder(num_classes, backbone, BatchNorm, ratio, input_heads)