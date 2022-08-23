# -*- coding: utf-8 -*-
# @Organization  : TMT
# @Author        : Cuong Tran
# @Time          : 8/18/2022

from functools import partial
import torch
from torch import nn, Tensor
from typing import List
from src.models.modules import LayerNorm2d, StochasticDepth, Permute, Conv2dNormActivation


class Block(nn.Module):
    def __init__(self, dim, layer_scale, stochastic_depth_prob):
        """
        :param in_planes:
        :param out_planes:
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNN(nn.Module):
    def __init__(self, stochastic_depth_prob=0.1, layer_scale=1e-6, dim=[32, 64, 128, 256]):
        """
              3 96 192 384 768
        """
        super().__init__()
        norm_layer = partial(LayerNorm2d, eps=1e-6)
        layers: List[nn.Module] = []
        firstconv_output_channels = dim[0]
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )
        total_stage_blocks = 4
        stage_block_id = 0
        block = Block
        for i in range(4):
            stage: List[nn.Module] = []
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            stage.append(block(dim[i], layer_scale, sd_prob))
            stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i < 3:
                layers.append(
                    nn.Sequential(
                        norm_layer(dim[i]),
                        nn.Conv2d(dim[i], dim[i+1], kernel_size=2, stride=2),
                    )
                )
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            norm_layer(dim[-1]), nn.Flatten(1), nn.Linear(dim[-1], 62)
        )

    def forward(self, x: Tensor):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = self.classifier(x)
        return x


if __name__ == '__main__':
    net = CNN()
    print(net)
    x = torch.rand((1, 3, 128, 1024))
    y = net(x)
    print(y.shape)


