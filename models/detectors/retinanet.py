import torch
import torch.nn as nn
import torchvision.transforms.functional as T
from torchvision.transforms import ToTensor, Lambda, Normalize, Compose, Resize
from ..transforms import transform, scale_factor, scale_size
from ..utils import bbox2result
from PIL import Image


class RetinaNet(nn.Module):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head):

        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.bbox_head = bbox_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.bbox_head(x)

    def detect(self, image):

        assert isinstance(image, str) or isinstance(image, Image.Image)

        image = Image.open(image) if isinstance(image, str) else image
        
        factor = scale_factor(image.size, scale=(1333, 800))
        scaled_size = scale_size(image.size, factor)

        results = self(transform(image, scaled_size).unsqueeze(0))

        bbox_list = self.bbox_head.get_bboxes(
            *results, [scaled_size], [factor], rescale=True)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes + 1)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results[0]
