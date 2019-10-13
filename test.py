from models.backbones import ResNet
from models.necks import FPN
from models.heads import RetinaHead
from models.detectors import RetinaNet
from models.utils import nms
import torch
import torch.nn as nn
from PIL import Image
import arrow
import cv2
import numpy as np


def strip(name):
    '''a.b.c -> a.b'''
    return '.'.join(name.split('.')[1:])

checkpoint_file = 'latest.pth'
test_image = 'test.jpg'

if __name__ == "__main__":
    deserialized = torch.load(checkpoint_file, map_location='cpu')['state_dict']
    backbone_state_dict = { strip(k):v for k,v in deserialized.items() if 'backbone' in k}
    resnet = ResNet(50)
    resnet.load_state_dict(backbone_state_dict)

    head_state_dict = {strip(k):v for k,v in deserialized.items() if 'head' in k}

    fpn = FPN( 
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5)
    
    retina_head = RetinaHead(
        num_classes=19,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0]
    )

    retina_head.load_state_dict(head_state_dict)

    fpn_state_dict = { strip(k):v for k,v in deserialized.items() if 'neck' in k}
    fpn.load_state_dict(fpn_state_dict)

    retina = RetinaNet(resnet, fpn, retina_head).eval()
    retina.load_state_dict(deserialized)

    retina = torch.quantization.quantize_dynamic(retina, dtype=torch.qint8)

    with torch.no_grad():
        start = arrow.now()
        result = retina.detect(Image.open(test_image))
        stop = arrow.now()
        print("Inference in: {}".format(stop-start))
    print(len(result))
    res = []
    for r in result[18]:
        if r[-1] >= .3:
            res.append(list(map(int, r[0:-1].astype(dtype=np.int).tolist())))

    if len(res) > 0:

        im = cv2.imread(test_image)
        for r in res:
            cv2.rectangle(im, (r[0], r[1]), (r[2], r[3]), (255, 255, 0), 3)
        
        cv2.imwrite('result.jpg', im)


    print(result)
    