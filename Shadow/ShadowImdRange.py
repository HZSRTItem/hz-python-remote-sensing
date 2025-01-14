# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowImdRange.py
@Time    : 2023/7/4 20:12
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ImdRange
-----------------------------------------------------------------------------"""
import numpy as np

from SRTCodes.NumpyUtils import reHist0

eps = 0.0000001


def reHist(d, ratio=0.001):
    d0, d1 = [], []
    for i in range(d.shape[0]):
        d_i = d[i]
        print(i)
        d00, d10 = reHist0(d_i, ratio)
        d0.append(d00)
        d1.append(d10)
    return np.array([d0, d1]).T


def plotImageHist(imd: np.array, bands=None, labels=None, kw=None):
    """ 绘制图像的波段数据分布图

    :param imd: 图像数据
    :param bands: 绘制的波段
    :param labels: 显示的标签
    :return:
    """
    if kw is None:
        if bands is None:
            bands = [i for i in range(imd.shape[0])]
        if labels is None:
            labels = ["Band " + str(i + 1) for i in range(imd.shape[0])]
    else:
        bands = list(kw.keys())
        labels = list(kw.values())

    print(bands)
    print(labels)

    d_range = {}

    for i, i_band in enumerate(bands):
        d = imd[i_band]
        print(labels[i], np.min(d), np.max(d))
        d_range0 = [np.min(d), np.max(d)]
        if i >= 6:
            d = 10 * np.log10(d + eps)

        # h, bin_edges = np.histogram(d, bins=256)
        # plt.plot(bin_edges[:-1], h, label=labels[i])

        d0, d1 = reHist0(d, ratio=0.01)
        d_range0.append(d0)
        d_range0.append(d1)
        d0, d1 = reHist0(d, ratio=0.02)
        d_range0.append(d0)
        d_range0.append(d1)

        print(d0, d1)

        # d = np.clip(d, d0, d1)
        # h, bin_edges = np.histogram(d, bins=256)
        # plt.plot(bin_edges[:-1], h, label=labels[i] + " reHist")

        d_range[labels[i]] = d_range0
        # plt.legend()
        # plt.show()

    print(d_range)

    # plt.legend()
    # plt.show()


def main():
    d = np.load(r"F:\ProjectSet\Shadow\QingDao\Image\stack2.npy")
    print(np.where(np.isnan(d)))

    plotImageHist(d, kw={
        0: "Blue",
        1: "Green",
        2: "Red",
        3: "NIR",
        4: "NDVI",
        5: "NDWI",
        6: "AS_VV",
        7: "AS_VH",
        8: "AS_C11",
        # 9: "AS_C12_imag",
        # 10: "AS_C12_real",
        11: "AS_C22",
        12: "AS_Lambda1",
        13: "AS_Lambda2",
        14: "DE_VV",
        15: "DE_VH",
        16: "DE_C11",
        # 17: "DE_C12_imag",
        # 18: "DE_C12_real",
        19: "DE_C22",
        20: "DE_Lambda1",
        21: "DE_Lambda2",
        # 22: "angle_AS",
        # 23: "angle_DE",
    })

    region = {
        "Blue": [1.0, 17824.0, 299.76996, 2397.184, 330.925, 1989.7163],
        "Green": [1.0, 16944.0, 345.83414, 2395.735, 377.25348, 1991.8011],
        "Red": [1.0, 16328.0, 177.79654, 2726.7026, 187.44577, 2310.2266],
        "NIR": [1.0, 15768.0, 87.66086, 3498.4321, 94.79853, 3232.0154],
        "NDVI": [-0.99866754, 0.99967706, -0.52984273, 0.7594785, -0.49883807, 0.7294391],
        "NDWI": [-0.9997585, 0.99783784, -0.6825941, 0.7639994, -0.65543485, 0.7393596],
        "AS_VV": [2.6482724e-06, 2652.358, -24.609674, 5.9092603, -23.272255, 3.3048003],
        "AS_VH": [6.6989304e-07, 1503.2048, -31.865038, -5.2615275, -30.63717, -7.3372407],
        "AS_C11": [0.0, 10667.435, -22.61998, 5.8634768, -21.73843, 3.1797814],
        "AS_C22": [0.0, 1821.584, -28.579813, -5.2111626, -27.579557, -7.2275467],
        "AS_Lambda1": [0.0, 11078.178, -21.955856, 6.124724, -21.138475, 3.325528],
        "AS_Lambda2": [0.0, 56.4261, -29.869734, -8.284683, -28.86374, -9.812354],
        "DE_VV": [1.0674944e-05, 2914.7483, -27.851603, 5.094706, -26.58056, 2.3995962],
        "DE_VH": [9.122703e-07, 474.4171, -35.427082, -5.4092093, -34.139236, -7.5289955],
        "DE_C11": [0.00016639428, 9388.018, -26.245598, 4.9907513, -25.351894, 2.3544536],
        "DE_C22": [2.0709329e-05, 689.79944, -32.04232, -5.322515, -31.136454, -7.437061],
        "DE_Lambda1": [0.00040819668, 9389.82, -25.503738, 5.2980003, -24.81052, 2.5125024],
        "DE_Lambda2": [1.733005e-05, 51.451904, -33.442368, -8.68537, -32.444366, -10.406565]
    }

    pass


if __name__ == "__main__":
    main()
