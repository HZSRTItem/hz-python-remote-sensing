# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : DenseNet.py
@Time    : 2024/7/21 20:12
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of DenseNet
https://arxiv.org/abs/1608.06993
https://blog.csdn.net/m0_74055982/article/details/137960751
-----------------------------------------------------------------------------"""

from collections import OrderedDict
from typing import Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from DeepLearning.Backbone import DenseBlock as _DenseBlock

class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
            self,
            growth_rate: int = 32,
            block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
            num_init_features: int = 64,
            bn_size: int = 4,
            drop_rate: float = 0,
            num_classes: int = 1000,
            memory_efficient: bool = False,
            in_channels=3,
    ) -> None:

        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            OrderedDict([
                ("conv0", nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
                ("norm0", nn.BatchNorm2d(num_init_features)),
                ("relu0", nn.ReLU(inplace=True)),
                # ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
            ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _densenet(growth_rate: int, block_config: Tuple[int, int, int, int], num_init_features: int,
              num_classes: int = 1000, in_channels=3,
              **kwargs: Any, ) -> DenseNet:
    model = DenseNet(growth_rate, block_config, num_init_features,
                     num_classes=num_classes, in_channels=in_channels, **kwargs)
    return model


def densenet121(num_classes: int = 1000, in_channels=3, ) -> DenseNet:
    r"""Densenet-121 model from`Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_. """
    return _densenet(32, (6, 12, 24, 16), 64, num_classes=num_classes, in_channels=in_channels)


def densenet161(num_classes: int = 1000, in_channels=3, ) -> DenseNet:
    r"""Densenet-161 model from`Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_. """
    return _densenet(48, (6, 12, 36, 24), 96, num_classes=num_classes, in_channels=in_channels)


def densenet169(num_classes: int = 1000, in_channels=3, ) -> DenseNet:
    r"""Densenet-169 model from`Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_. """
    return _densenet(32, (6, 12, 32, 32), 64, num_classes=num_classes, in_channels=in_channels)


def densenet201(num_classes: int = 1000, in_channels=3, ) -> DenseNet:
    r"""Densenet-201 model from`Densely Connected Convolutional Networks <https://arxiv.org/abs/1608.06993>`_. """
    return _densenet(32, (6, 12, 48, 32), 64, num_classes=num_classes, in_channels=in_channels)


def main():
    growth_rate = 32
    in_channels = 3
    block_config = (6, 12, 24, 16)
    num_init_features = 64

    bn_size: int = 4
    drop_rate: float = 0
    num_classes: int = 1000
    memory_efficient: bool = False

    # First convolution
    features = [
        ("conv0", nn.Conv2d(in_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
        ("norm0", nn.BatchNorm2d(num_init_features)),
        ("relu0", nn.ReLU(inplace=True)),
        ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]

    # Each denseblock
    num_features = num_init_features
    for i, num_layers in enumerate(block_config):
        block = _DenseBlock(
            num_layers=num_layers,
            num_input_features=num_features,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate,
            memory_efficient=memory_efficient,
        )
        features.append(("denseblock%d" % (i + 1), block))
        num_features = num_features + num_layers * growth_rate
        if i != len(block_config) - 1:
            trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
            features.append(("transition%d" % (i + 1), trans))
            num_features = num_features // 2

    # Final batch norm
    features.append(("norm5", nn.BatchNorm2d(num_features)))

    # Linear layer
    # features.append(("classifier", nn.Linear(num_features, num_classes)))

    x = torch.rand(2, 3, 224, 224)
    print("{:<16} {} ".format("init", x.shape))
    for name, mod in features:
        x = mod(x)
        print("{:<16} {} ".format(name, x.shape))





if __name__ == "__main__":
    main()
