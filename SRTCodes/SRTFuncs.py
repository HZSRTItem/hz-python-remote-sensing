import sys

import torch
from torch import nn

sys.path.append(r"F:\PyCodes")
# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTFuncs.py
@Time    : 2023/11/3 9:59
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTFuncs
-----------------------------------------------------------------------------"""

import numpy as np
from scipy.ndimage import uniform_filter

from SRTCodes.GDALRasterIO import readGEORaster, saveGEORaster
from Shadow.ShadowGeoDraw import _10log10


class TNet(nn.Module):

    def __init__(self):
        super(TNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 2, 3, 1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


def main():
    x = torch.randn((1, 3, 128, 128))
    mod = TNet()
    out_x = mod(x)

    pass


def method_name1():
    # Define
    def lee_filter(image, window_size):
        """ Lee filter function """
        # Calculate the local mean using a convolution
        local_mean = uniform_filter(image, (window_size, window_size))

        # Calculate the local variance
        local_variance = uniform_filter(image ** 2, (window_size, window_size)) - local_mean ** 2

        # Estimate the noise variance
        noise_variance = local_variance.mean()

        # Calculate the filtered image
        filtered_image = local_mean + (image - local_mean) * np.minimum(
            noise_variance / (local_variance + noise_variance),
            1)

        return filtered_image

    # Load your SAR image as a NumPy array
    # Replace 'your_image_array' with your SAR image array
    # Example:
    # your_image_array = np.array([[...]])
    your_image_array = readGEORaster(r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_envi.dat", band_list=["AS_C11"])
    print(your_image_array.shape)
    # Set the window size for the filter (you can adjust this parameter)
    window_size = 7
    # Apply the Lee filter to the SAR image
    filtered_sar_image = lee_filter(your_image_array, window_size)
    print(filtered_sar_image.shape)
    saveGEORaster(_10log10(filtered_sar_image), r"F:\ProjectSet\Shadow\BeiJing\Image\Temp\tmp35")


if __name__ == "__main__":
    main()
