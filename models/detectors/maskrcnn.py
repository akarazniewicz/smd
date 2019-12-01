import torch
import torchvision
import torchvision.models.detection.mask_rcnn as mask_rcnn
import torchvision.transforms as transforms
from PIL import Image
from ..transforms import transform, scale_factor, scale_size
from .fasterrcnn import FasterRCNNMapper


class MaskRCNN(mask_rcnn.MaskRCNN):
    ''' 
        For MaskRCNN we simply use torchvision implementation with weight transformation 
        from mmdetection.
    '''

    def __init__(self, backbone, num_classes):
        super().__init__(backbone, num_classes)

    def detect(self, image):
        assert isinstance(image, str) or isinstance(image, Image.Image)
        image = Image.open(image) if isinstance(image, str) else image
        transform = transforms.ToTensor()
        return self(transform(image).unsqueeze(0))


class MaskRCNNMapper(FasterRCNNMapper):
    '''
        Maps mmdetection state dict to torchvision model.
    '''

    def m_to_t(self, k):
        tokens = k.split('.')
        if 'mask_head.convs' in k:
            return self.dotty('roi_heads.mask_head.mask_fcn' + str(1+int(tokens[-3])), tokens[-1])
        elif 'mask_head.upsample' in k:
            return self.dotty('roi_heads.mask_predictor.conv5_mask', tokens[-1])
        elif 'mask_head.conv_logits' in k:
            return self.dotty('roi_heads.mask_predictor.mask_fcn_logits', tokens[-1])
        else:
            return super().m_to_t(k)

    