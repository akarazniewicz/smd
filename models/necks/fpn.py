import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import ConvModule


class FPN(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        num_outs,
        activation=None
    ):
        super().__init__()
        assert(isinstance(in_channels, list))
        assert(isinstance(activation, nn.Module) or activation is None)

        self.in_channels = in_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.end_level = self.num_ins - 1

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.activation = activation

        for i in range(1, self.num_ins):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                activation=activation
            )
            self.lateral_convs.append(l_conv)

            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                activation=activation
            )
            self.fpn_convs.append(fpn_conv)

        extra_levels = num_outs - self.num_ins + 1
        if extra_levels > 1:
            for i in range(extra_levels):
                if i == 0:
                    in_channels = self.in_channels[self.num_ins - 1]
                else:
                    in_channels = out_channels

                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    activation=activation
                )

                self.fpn_convs.append(extra_fpn_conv)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        laterals = [lateral_conv(inputs[i+1])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        outs = [self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels)]

        orig = inputs[self.num_ins - 1]
        outs.append(self.fpn_convs[used_backbone_levels](orig))

        for i in range(used_backbone_levels + 1, self.num_outs):
            outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
