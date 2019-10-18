import torch
from ..backbones import ResNet
from ..necks import FPN
from ..heads import RetinaHead
from .retinanet import RetinaNet


def create_detector(name, number_of_classes, pretrained, device='cpu'):

    architectures = {
        'retinanet_r50_fpn_1x': RetinaNet(
            ResNet(50),
            FPN(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
            RetinaHead(
                num_classes=number_of_classes,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                octave_base_scale=4,
                scales_per_octave=3,
                anchor_ratios=[0.5, 1.0, 2.0],
                anchor_strides=[8, 16, 32, 64, 128],
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]
            )),
        'retinanet_r101_fpn_1x': RetinaNet(
            ResNet(101),
            FPN(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5),
            RetinaHead(
                num_classes=number_of_classes,
                in_channels=256,
                stacked_convs=4,
                feat_channels=256,
                octave_base_scale=4,
                scales_per_octave=3,
                anchor_ratios=[0.5, 1.0, 2.0],
                anchor_strides=[8, 16, 32, 64, 128],
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]
            ))
    }

    model = architectures.get(name)

    if model is not None:
        state_dict = torch.load(pretrained, map_location=device)['state_dict']
        model.load_state_dict(state_dict)
        return model.eval()
    else:
        raise ValueError("Invalid model name. Supported models: {}".format(architectures.keys)) 
