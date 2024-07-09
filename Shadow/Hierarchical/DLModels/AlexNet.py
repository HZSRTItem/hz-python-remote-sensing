# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : AlexNet.py
@Time    : 2024/7/9 16:26
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of AlexNet
-----------------------------------------------------------------------------"""
import torch
from torch import nn

from Shadow.Hierarchical.DLModels.SHH2Training import training, loadDS, imdcing


class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000, in_ch=3, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def main():
    n_ch = 18
    model = AlexNet(4, n_ch)
    win_size = (21, 21)
    x = torch.rand(10, n_ch, *win_size)
    out_x = model(x)

    model = nn.Sequential(
        nn.Conv2d(n_ch, n_ch, 3, 1, 1),
        nn.Flatten(start_dim=1),
        nn.Linear(21 * 21 * n_ch, 4),
    )

    train_ds, test_ds = loadDS("qd", win_size=win_size, read_size=(21, 21))
    training(model, train_ds, test_ds, epochs=2)
    imdcing("qd", model, win_size)

    pass


if __name__ == "__main__":
    main()
