# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHDLNetThreeBranches.py
@Time    : 2023/11/29 10:13
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHDLNet_ThreeBranches
-----------------------------------------------------------------------------"""
import os.path
from typing import Union

import torch
from torch import nn


def netConv2dNormAct(in_channel,
                     out_channels: list = None, mul_channels: list = None,
                     act=nn.Sigmoid,
                     norm=nn.BatchNorm2d,
                     kernel_size: Union[int, tuple] = 3,
                     stride: Union[int, tuple] = 1,
                     padding: Union[str, int, tuple] = 1,
                     pools: list = None,
                     is_ret_list=False
                     ):
    """ Conv2d + Norm + Act
        for net in netConv2dNormAct():
            x = net(x)
            ...
    """
    if out_channels is None:
        out_channels = []
    if mul_channels is None:
        mul_channels = []
    if not out_channels:
        out_channels = [in_channel * mul_c for mul_c in mul_channels]
    if pools is None:
        pools = [None for _ in range(len(out_channels))]
    if (not out_channels) and (not mul_channels):
        raise Exception("Func: netConv2dBnAct. Can not find channels.")
    out_channels = [in_channel] + out_channels
    nets = []
    for i in range(len(out_channels) - 1):
        in_c, out_c = out_channels[i], out_channels[i + 1]
        nets.append(nn.Conv2d(in_channels=in_c, out_channels=out_c,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              padding_mode="reflect"))
        nets.append(norm(out_c))
        nets.append(act())
        if pools[i] is not None:
            pool = pools[i]
            if pool == "max_pool2d":
                nets.append(nn.MaxPool2d(2))
    # printLines(nets, is_line_number=True)
    if is_ret_list:
        return nets
    else:
        return nn.Sequential(*tuple(nets))


class SHDLNet_SimpleThreeBranches(nn.Module):
    """
    三分支输入的模型
    第一分支：光学分支
    第二分支：AS分支
    第三分支：DE分支
    """

    def __init__(self, optical_channels=4, as_channels=2, de_channels=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.this_file = os.path.abspath(__file__)

        self.optical_net = netConv2dNormAct(
            in_channel=optical_channels, mul_channels=[2, 4, 8, 16], pools=[None, None, "max_pool2d", None])
        self.as_net = netConv2dNormAct(
            in_channel=as_channels, mul_channels=[2, 4, 8, 16], pools=[None, None, "max_pool2d", None])
        self.de_net = netConv2dNormAct(
            in_channel=de_channels, mul_channels=[2, 4, 8, 16], pools=[None, None, "max_pool2d", None])

        # self.cna = netConv2dNormAct(
        #     in_channel=128, mul_channels=[2, 4, 8, 8], pools=[None, None, None, None])
        # self.max_pool = nn.MaxPool2d((2, 2), stride=2)
        #
        # self.conv_end1 = nn.Conv2d(1024, 1024, (2, 2), stride=1)
        # self.bn_end1 = nn.BatchNorm2d(1024)

        self.fc1 = nn.LazyLinear(100)
        self.fc_act1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 20)
        self.fc_act2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 2)

    def forward(self, x):
        x_optical, x_as, x_de = x[:, :4], x[:, 4:6], x[:, 6:8]
        x_optical = self.optical_net(x_optical)
        x_as = self.as_net(x_as)
        x_de = self.de_net(x_de)
        x = torch.concat([x_optical, x_as, x_de], dim=1)
        # x = self.cna(x)
        # x = self.conv_end1(x)
        # x = self.bn_end1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc_act1(x)
        x = self.fc2(x)
        x = self.fc_act2(x)
        x = self.fc3(x)
        return x


class SHDLNet_Cross(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, y: torch.Tensor, lamd=1):
        logt = self.sigmoid_act(x)
        y = y.view((-1, 1))
        loss = -torch.mean(lamd * (y * torch.log(logt) + (1 - y) * torch.log(1 - logt)))
        return loss


def main():
    # model = SHDLNet_SimpleThreeBranches()
    # x = torch.rand((32, 8, 9, 9))
    # print(x.shape)
    # x = model(x[:, :4, :, :], x[:, 4:6, :, :], x[:, 6:8, :, :])
    # print(x.shape)
    pass


if __name__ == "__main__":
    main()
