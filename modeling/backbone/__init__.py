from modeling.backbone import resnet, xception, drn, mobilenet, resnet_adv, xception_adv

def build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=False):
    if backbone == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm, pretrained=False)
    elif backbone == 'resnet_adv':
        return resnet_adv.ResNet101(output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    elif backbone == 'resnet_condconv':
        return resnet_condconv.ResNet101(output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'xception_adv':
        return xception_adv.AlignedXception(output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError