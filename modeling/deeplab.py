import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone


class DeepLabMultiInput(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False, input_dim=3, ratio=1, pretrained=False):
        super(DeepLabMultiInput, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        
        self.backbone1 = build_backbone(backbone, output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained) # RGB
        self.aspp1 = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone2 = build_backbone(backbone, output_stride, BatchNorm, input_dim=2) # aolp
        self.aspp2 = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone3 = build_backbone(backbone, output_stride, BatchNorm, input_dim=1) # dolp
        self.aspp3 = build_aspp(backbone, output_stride, BatchNorm)
        self.backbone4 = build_backbone(backbone, output_stride, BatchNorm, input_dim=1) # nir
        self.aspp4 = build_aspp(backbone, output_stride, BatchNorm)

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, ratio, input_heads=4)

        self.freeze_bn = freeze_bn
    def forward(self, input1, input2=None, input3=None, input4=None, mask=None):

        x1, low_level_feat1 = self.backbone1(input1)
        x1 = self.aspp1(x1)
        # AoLP
        if input2 is not None:
            x2, low_level_feat2 = self.backbone2(input2)
            #x2=x2[0]
            x2 = self.aspp2(x2)
        else:
            x2 = torch.zeros_like(x1)
            low_level_feat2 = torch.zeros_like(low_level_feat1)
        # DoLP
        if input3 is not None:
            x3, low_level_feat3 = self.backbone3(input3)
            #x3=x3[0]
            x3 = self.aspp3(x3)
        else:
            x3 = torch.zeros_like(x1)
            low_level_feat3 = torch.zeros_like(low_level_feat1)
        # NIR
        if input4 is not None:
            x4, low_level_feat4 = self.backbone4(input4)
            #x4=x4[0]
            x4 = self.aspp4(x4)
        else:
            x4 = torch.zeros_like(x1)
            low_level_feat4 = torch.zeros_like(low_level_feat1)

        x = torch.cat([x1,x2,x3,x4],dim=1) 

        low_level_feat = torch.cat([low_level_feat1,low_level_feat2,low_level_feat3,low_level_feat4],dim=1)
        x = self.decoder(x, low_level_feat, mask)

        x = F.interpolate(x, size=input1.size()[2:], mode='bilinear', align_corners=True)


        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone1,self.backbone2,self.backbone3,self.backbone4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                #print(p)
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                               # print(p)
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp1, self.aspp2, self.aspp3, self.aspp4, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():

                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                
                                yield p
                else:
            
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1],nn.Linear):
                        for p in m[1].parameters():
                            #if m[0].split('.')[0]=='condconv':
                                #continue
                            if p.requires_grad:
                                #print(m[0])
                                yield p
                    if m[0]=='gamma':
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
    '''
    def get_100x_lr_params(self):
        modules = [self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():

                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                #print(m[0])
                                yield p
                else:
                    if m[0].split('.')[0]=='condconv':
                        for p in m[1].parameters():
                            if p.requires_grad:
                                #print(m[0])
                                yield p
        #for  m, parameters in modules[4].named_parameters():
            #for p in parameters:
                #if p.requires_grad:
                    #yield p
    '''

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


