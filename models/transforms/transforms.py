import numpy as np
from PIL import Image
import torchvision.transforms.functional as T
from torchvision.transforms import Resize, Compose, Lambda, ToTensor, Normalize

def scale_factor(size, scale):
    w, h = size
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    return min(max_long_edge / max(h, w),
               max_short_edge / min(h, w))


def scale_size(size, scale):
    w, h = size
    return int(h * float(scale) + 0.5), int(w * float(scale) + 0.5)


def padding(image, divisor=32):
    w, h = image.size
    w_pad = divisor - (w % divisor)
    h_pad = divisor - (h % divisor)
    e = 0, 0, w_pad if w_pad < divisor else 0, h_pad if h_pad < divisor else 0
    return e


def transform(image, scaled_size):

    assert isinstance(image, Image.Image)

    transforms = Compose([
        Resize(scaled_size),
        Lambda(lambda i: T.pad(i, padding(i))),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])

    return transforms(image)
