import torch
from torch import nn
from ..utils import conv_layer, norm_layer

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 strinde=1,
                 dilation=1,
                 downsample=None):

        super().__init__()

        self.norm1_name, norm1 = norm_layer(planes=planes, postfix=1)
        self.norm2_name, norm2 = norm_layer(planes=planes, postfix=2)

        self.conv1 = conv_layer(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv_layer(planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)
        self.downsample

        self.relu = nn.ReLU(inplace=True)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None):

        super().__init__()

        self.norm1_name, norm1 = norm_layer(planes, postfix=1)
        self.norm2_name, norm2 = norm_layer(planes, postfix=2)
        # expansion by 4
        self.norm3_name, norm3 = norm_layer(planes * self.expansion, postfix=3)

        self.conv1 = conv_layer(
            inplanes, planes, stride=1, kernel_size=1, bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = conv_layer(
            planes, planes, stride=stride, kernel_size=3, bias=False, padding=dilation, dilation=dilation)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = conv_layer(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return self.relu(out)


def resnet_layer(block, inplanes, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv_layer(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            norm_layer(planes * block.expansion)[1],
        )

    layers = []
    layers.append(block(
        inplanes=inplanes,
        planes=planes,
        stride=stride,
        dilation=dilation,
        downsample=downsample)
    )

    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(
            inplanes=inplanes,
            planes=planes,
            stride=1, 
            dilation=dilation)
        )

    return nn.Sequential(*layers)


class ResNet(nn.Module):

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3)
                 ):

        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.strides = strides
        self.dilations = dilations
        self.out_indices = out_indices
        self.inplanes = 64

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]

        self.conv1 = conv_layer(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False)
        self.norm1_name, norm1 = norm_layer(64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = 64 * 2**i
            res_layer = resnet_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i+1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2**(
            len(self.stage_blocks) - 1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
