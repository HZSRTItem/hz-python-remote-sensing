# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : IEEEGRSSMain.py
@Time    : 2024/1/8 21:01
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of IEEEGRSSMain
-----------------------------------------------------------------------------"""
import os

import numpy as np
import torch
import torch.nn as nn
from osgeo import gdal
from torch.utils.data import Dataset, DataLoader

from SRTCodes.Utils import getfilenamewithoutext


def readData(filename):
    ds = gdal.Open(filename)
    d = ds.ReadAsArray()
    return d


# 定义U-Net模型
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.encoder_conv1 = nn.Conv2d(2, 64, 3, padding=1)
        self.encoder_relu1 = nn.ReLU(inplace=True)

        self.encoder_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.encoder_relu2 = nn.ReLU(inplace=True)

        self.encoder_max_pool2d1 = nn.MaxPool2d(2, stride=2)

        self.encoder_conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.encoder_relu3 = nn.ReLU(inplace=True)

        self.encoder_conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.encoder_relu4 = nn.ReLU(inplace=True)

        self.encoder_max_pool2d2 = nn.MaxPool2d(2, stride=2)

        self.encoder_conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.encoder_relu5 = nn.ReLU(inplace=True)

        self.encoder_conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.encoder_relu6 = nn.ReLU(inplace=True)

        self.encoder_max_pool2d3 = nn.MaxPool2d(2, stride=2)

        self.encoder_conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.encoder_relu7 = nn.ReLU(inplace=True)

        self.encoder_conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.encoder_relu8 = nn.ReLU(inplace=True)

        self.encoder_max_pool2d4 = nn.MaxPool2d(2, stride=2)

        self.decoder_conv_t2d1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder_relu01 = nn.ReLU(inplace=True)

        self.decoder_conv1 = nn.Conv2d(256, 256, 3, padding=1)
        self.decoder_relu1 = nn.ReLU(inplace=True)

        self.decoder_conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.decoder_relu2 = nn.ReLU(inplace=True)

        self.decoder_conv_t2d2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder_relu02 = nn.ReLU(inplace=True)

        self.decoder_conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.decoder_relu3 = nn.ReLU(inplace=True)

        self.decoder_conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.decoder_relu4 = nn.ReLU(inplace=True)

        self.decoder_conv_t2d3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_relu03 = nn.ReLU(inplace=True)

        self.decoder_conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.decoder_relu5 = nn.ReLU(inplace=True)

        self.decoder_conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.decoder_relu6 = nn.ReLU(inplace=True)

        # 输出层
        self.output = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x = self.encoder_conv1(x)
        x = self.encoder_relu1(x)
        x = self.encoder_conv2(x)
        x = self.encoder_relu2(x)
        x = self.encoder_max_pool2d1(x)
        x = self.encoder_conv3(x)
        x = self.encoder_relu3(x)
        x = self.encoder_conv4(x)
        x = self.encoder_relu4(x)
        x = self.encoder_max_pool2d2(x)
        x = self.encoder_conv5(x)
        x = self.encoder_relu5(x)
        x = self.encoder_conv6(x)
        x = self.encoder_relu6(x)
        x = self.encoder_max_pool2d3(x)
        x = self.encoder_conv7(x)
        x = self.encoder_relu7(x)
        x = self.encoder_conv8(x)
        x = self.encoder_relu8(x)
        x = self.encoder_max_pool2d4(x)

        x = self.decoder_conv_t2d1(x)
        x = self.decoder_relu01(x)
        x = self.decoder_conv1(x)
        x = self.decoder_relu1(x)
        x = self.decoder_conv2(x)
        x = self.decoder_relu2(x)
        x = self.decoder_conv_t2d2(x)
        x = self.decoder_relu02(x)
        x = self.decoder_conv3(x)
        x = self.decoder_relu3(x)
        x = self.decoder_conv4(x)
        x = self.decoder_relu4(x)
        x = self.decoder_conv_t2d3(x)
        x = self.decoder_relu03(x)
        x = self.decoder_conv5(x)
        x = self.decoder_relu5(x)
        x = self.decoder_conv6(x)
        x = self.decoder_relu6(x)

        x = self.output(x)
        return x


class IEEEGRSSDataset(Dataset):

    def __init__(self, init_dirname, ds_type="train"):
        super(IEEEGRSSDataset, self).__init__()
        self.init_dirname = init_dirname
        self.ds_type = ds_type
        self.x_dirname = None
        self.y_dirname = None
        self.fns = []
        self.init()

    def init(self):
        if self.ds_type == "train":
            self.x_dirname = os.path.join(self.init_dirname, "train", "images")
            self.y_dirname = os.path.join(self.init_dirname, "train", "labels")
        elif self.ds_type == "val":
            self.x_dirname = os.path.join(self.init_dirname, "val", "images")
        self.fns = []
        if self.x_dirname is not None:
            for fn in os.listdir(self.x_dirname):
                if os.path.splitext(fn)[1] == ".tif":
                    fn = getfilenamewithoutext(fn)
                    x_fn = os.path.join(self.x_dirname, fn + ".tif")
                    y_fn = os.path.join(self.y_dirname, fn + ".png")
                    if not os.path.isfile(x_fn):
                        print("Can not find X file {0}".format(x_fn))
                    if not os.path.isfile(y_fn):
                        print("Can not find Y file {0}".format(y_fn))
                    if os.path.isfile(x_fn) and os.path.isfile(y_fn):
                        self.fns.append(fn)

        self.fns.sort()

    def __getitem__(self, item):
        fn = str(item)
        if fn not in self.fns:
            return None
        x_fn = os.path.join(self.x_dirname, "{0}.tif".format(fn))
        y_fn = os.path.join(self.y_dirname, "{0}.png".format(fn))
        x, y = readData(x_fn), readData(y_fn)
        x = x[:2]
        x = np.clip(x, 0, 5000)
        x = x / 5000
        return x, y

    def __len__(self):
        return len(self.fns)


def main():

    train_dataset = IEEEGRSSDataset(r"G:\ImageData\Track1\Track1")
    test_dataset = IEEEGRSSDataset(r"G:\ImageData\Track1\Track1")

    batch_size = 8
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 实例化模型和损失函数
    model = UNet(2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        n = 0
        for images, labels in train_loader:
            images = images.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if n % 16 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], N:{n}, Loss: {running_loss / len(train_loader)}")

            n += 1

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    # 测试模型
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.float().to(device)
            labels = labels.to(device)

            outputs = model(images)
            test_loss += criterion(outputs, labels).item()

        average_loss = test_loss / len(test_loader)
        print(f"Test Loss: {average_loss}")
    pass


if __name__ == "__main__":
    main()
