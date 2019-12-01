import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from ..backbones import ResNet
from ..necks import FPN
from ..heads import RetinaHead
from .retinanet import RetinaNet
from .maskrcnn import MaskRCNN, MaskRCNNMapper
from .fasterrcnn import FasterRCNN, FasterRCNNMapper


def create_detector(name, number_of_classes, pretrained, device='cpu'):

    architectures = {
        'retinanet_r50_fpn': (None,
                              RetinaNet(
                                  ResNet(50),
                                  FPN(
                                      in_channels=[
                                          256, 512, 1024, 2048],
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
                                      anchor_strides=[
                                          8, 16, 32, 64, 128],
                                      target_means=[.0, .0, .0, .0],
                                      target_stds=[1.0, 1.0, 1.0, 1.0]
                                  ))),
        'retinanet_r101_fpn': (None,
                               RetinaNet(
                                   ResNet(101),
                                   FPN(
                                       in_channels=[
                                           256, 512, 1024, 2048],
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
                                       anchor_strides=[
                                           8, 16, 32, 64, 128],
                                       target_means=[.0, .0, .0, .0],
                                       target_stds=[1.0, 1.0, 1.0, 1.0]
                                   ))),
        'mask_rcnn_r50_fpn': (MaskRCNNMapper(),
                              MaskRCNN(backbone=resnet_fpn_backbone('resnet50', pretrained=False),
                                       num_classes=number_of_classes
                                       )),
        'mask_rcnn_r101_fpn': (MaskRCNNMapper(),
                               MaskRCNN(backbone=resnet_fpn_backbone('resnet101', pretrained=False),
                                        num_classes=number_of_classes
                                        )),
        'faster_rcnn_r50_fpn': (FasterRCNNMapper(),
                                FasterRCNN(backbone=resnet_fpn_backbone('resnet50', pretrained=False),
                                           num_classes=number_of_classes
                                           )),
        'faster_rcnn_r101_fpn': (FasterRCNNMapper(),
                                 FasterRCNN(backbone=resnet_fpn_backbone('resnet101', pretrained=False),
                                            num_classes=number_of_classes
                                            ))
    }

    arch = architectures.get(name)

    if arch is not None:
        mapper, model = arch
        state_dict = torch.load(pretrained, map_location=device)['state_dict']
        model.load_state_dict(mapper(state_dict) if mapper else state_dict)
        return model.eval()
    else:
        raise ValueError(
            "Invalid model: `{}`. Supported models: {}".format(name, architectures.keys()))
