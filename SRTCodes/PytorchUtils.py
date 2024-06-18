# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : PytorchUtils.py
@Time    : 2024/5/31 17:25
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of PytorchUtils
-----------------------------------------------------------------------------"""
from torch import nn


def convBnAct(in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, act=nn.ReLU()):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        act,
    )

def main():
    pass


if __name__ == "__main__":
    main()
