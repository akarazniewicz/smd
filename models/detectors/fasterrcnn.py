import torch
import torchvision
import torchvision.models.detection.faster_rcnn as faster_rcnn
import torchvision.transforms as transforms
from PIL import Image
from ..transforms import transform, scale_factor, scale_size


class FasterRCNN(faster_rcnn.FasterRCNN):
    '''
        For Fast-RCNN we simply use torchvision FasterRCNN implementation
        and map mmdetection 'state_dict'.
    '''

    def __init__(self, backbone, num_classes=None):
        super().__init__(backbone, num_classes=num_classes)

    def detect(self, image):
        assert isinstance(image, str) or isinstance(image, Image.Image)
        image = Image.open(image) if isinstance(image, str) else image
        transform = transforms.ToTensor()
        return self(transform(image).unsqueeze(0))


class FasterRCNNMapper():
    '''
        Maps mmdetection state dict to torchvision model.
    '''

    def dotty(self, *elms):
        return '.'.join(elms)

    def m_to_t(self, k):
        tokens = k.split('.')
        if 'backbone' in k:
            return self.dotty('backbone.body', *tokens[1:])
        elif 'neck.lateral_convs' in k:
            # 'neck.lateral_convs.0.conv.weight' > 'backbone.fpn.inner_blocks.0.weight'
            return self.dotty('backbone.fpn.inner_blocks', tokens[-3], tokens[-1])
        elif 'neck.fpn_convs' in k:
            return self.dotty('backbone.fpn.layer_blocks', tokens[-3], tokens[-1])
        elif 'rpn_head.rpn_conv' in k:
            return self.dotty('rpn.head.conv', tokens[-1])
        elif 'rpn_head.rpn_cls' in k:
            return self.dotty('rpn.head.cls_logits', tokens[-1])
        elif 'rpn_head.rpn_reg' in k:
            return self.dotty('rpn.head.bbox_pred', tokens[-1])
        elif 'bbox_head.shared_fcs.0' in k:
            return self.dotty('roi_heads.box_head.fc6', tokens[-1])
        elif 'bbox_head.shared_fcs.1' in k:
            return self.dotty('roi_heads.box_head.fc7', tokens[-1])
        elif 'bbox_head.fc_cls' in k:
            return self.dotty('roi_heads.box_predictor.cls_score', tokens[-1])
        elif 'bbox_head.fc_reg' in k:
            return self.dotty('roi_heads.box_predictor.bbox_pred', tokens[-1])
        return k

    def __call__(self, state_dict):
        return {self.m_to_t(k): v for k, v in state_dict.items() if 'num_batches' not in k}

