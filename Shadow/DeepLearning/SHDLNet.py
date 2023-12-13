# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHDLNet.py
@Time    : 2023/12/1 20:22
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHDLNet
-----------------------------------------------------------------------------"""
import os

from torch import nn


class SHDLNet_Test(nn.Module):

    def __init__(self):
        super(SHDLNet_Test, self).__init__()
        self.this_file = os.path.abspath(__file__)

        self.conv1 = nn.Conv2d(8, 16, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()

        self.conv1_pool = nn.Conv2d(32, 32, 2, stride=2)
        self.sigmoid1 = nn.Sigmoid()

        self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(8, 8 * 6)
        self.fc1_act = nn.Sigmoid()
        self.fc2 = nn.Linear(8 * 6, 8 * 12)
        self.fc2_act = nn.Sigmoid()
        self.fc3 = nn.Linear(8 * 12, 8*6)
        self.fc3_act = nn.Sigmoid()

        self.fc_cate = nn.Linear(8*6, 2)

    def forward(self, x):
        x = x[:, :, 4, 4]

        x = self.fc1(x)
        x = self.fc1_act(x)

        x = self.fc2(x)
        x = self.fc2_act(x)

        x = self.fc3(x)
        x = self.fc3_act(x)

        x = self.fc_cate(x)

        return x


def main():
    pass


if __name__ == "__main__":
    main()
