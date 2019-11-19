import torch
import torchvision
import torchvision.models.detection.mask_rcnn as mask_rcnn
import torchvision.transforms as transforms
from PIL import Image
from ..transforms import transform, scale_factor, scale_size


class MaskRCNN(mask_rcnn.MaskRCNN):
    ''' 
        For MaskRCNN we simply use torchvision implementation with weight transformation 
        from mmdetection.
    '''

    def __init__(self, backbone, num_classes):

        super().__init__(
            backbone,
            num_classes)

    def detect(self, image):
        assert isinstance(image, str) or isinstance(image, Image.Image)
        image = Image.open(image) if isinstance(image, str) else image
        transform = transforms.ToTensor()
        return self(transform(image).unsqueeze(0))


class MaskRCNNMapper():
    '''
        Maps mmdetection state dict to torchvision model.
    '''

    def __call__(self, state_dict):

        def dotty(*elms):
            return '.'.join(elms)

        def m_to_t(k):

            tokens = k.split('.')
            if 'backbone' in k:
                return dotty('backbone.body', *k.split('.')[1:])
            elif 'neck.lateral_convs' in k:
                # 'neck.lateral_convs.0.conv.weight' > 'backbone.fpn.inner_blocks.0.weight'
                return dotty('backbone.fpn.inner_blocks', tokens[-3], tokens[-1])
            elif 'neck.fpn_convs' in k:
                return dotty('backbone.fpn.layer_blocks', tokens[-3], tokens[-1])
            elif 'rpn_head.rpn_conv' in k:
                return dotty('rpn.head.conv', tokens[-1])
            elif 'rpn_head.rpn_cls' in k:
                return dotty('rpn.head.cls_logits', tokens[-1])
            elif 'rpn_head.rpn_reg' in k:
                return dotty('rpn.head.bbox_pred', tokens[-1])
            elif 'bbox_head.shared_fcs.0' in k:
                return dotty('roi_heads.box_head.fc6', tokens[-1])
            elif 'bbox_head.shared_fcs.1' in k:
                return dotty('roi_heads.box_head.fc7', tokens[-1])
            elif 'bbox_head.fc_cls' in k:
                return dotty('roi_heads.box_predictor.cls_score', tokens[-1])
            elif 'bbox_head.fc_reg' in k:
                return dotty('roi_heads.box_predictor.bbox_pred', tokens[-1])
            elif 'mask_head.convs' in k:
                return dotty('roi_heads.mask_head.mask_fcn' + str(1+int(tokens[-3])), tokens[-1])
            elif 'mask_head.upsample' in k:
                return dotty('roi_heads.mask_predictor.conv5_mask', tokens[-1])
            elif 'mask_head.conv_logits' in k:
                return dotty('roi_heads.mask_predictor.mask_fcn_logits', tokens[-1])
            return k

        return {m_to_t(k): v for k, v in state_dict.items() if 'num_batches' not in k}
