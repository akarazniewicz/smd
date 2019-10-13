import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ..transform import Compose, Pad, Resize, MultiScaleFlipAug, Normalize, LoadImage, ImageToTensor, bbox2result


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
        transform = transforms.Compose([
            transforms.Scale((1333, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
        ])

        tr = Compose([
            LoadImage(),
            MultiScaleFlipAug(img_scale=(1333,800), flip=False, transforms=[
                Resize(keep_ratio=True),
                Normalize(**dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)),
                Pad(size_divisor=32),
                ImageToTensor(keys=['img'])
            ])
        ])

        pipeline = tr(dict(img='test.jpg'))
        #print(pipeline['img'][0].unsqueeze(0).shape)
        #print(transform(image).unsqueeze(0).permute(0,1,3,2).shape)

        results = self(pipeline['img'][0].unsqueeze(0))
        img_metas=[dict(img_shape=pipeline['img_shape'][0], scale_factor=pipeline['scale_factor'][0])]
        
        #results = self(transform(image).unsqueeze(0).permute(0,1,3,2))
        #img_metas=[dict(img_shape=(1333, 800), scale_factor=800/image.size.height)]
        
        cfg = dict(score_thr=0.05)
        bbox_list = results + (img_metas, cfg)
        bbox_list = self.bbox_head.get_bboxes(*bbox_list, rescale=True)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes + 1)
            for det_bboxes, det_labels in bbox_list
        ]

        return bbox_results[0]

   
