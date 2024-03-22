# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : PandasUtils.py
@Time    : 2024/1/29 10:57
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of PandasUtils
-----------------------------------------------------------------------------"""

import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave




def filterDF(df, *filters, **filter_maps):
    filter_dict = {}
    for f in filters:
        filter_dict[f[0]] = f[1]
    for f in filter_maps:
        filter_dict[f] = filter_maps[f]
    for k, f in filter_dict.items():
        df = df[df[k] == f]
    return df


def main():
    im = imread(r"F:\ProjectSet\Shadow\MkTu\Imdc\ShadowMkTuImdc3.jpg")
    print(type(im))  # ---><class 'numpy.ndarray'>
    print(im.shape)  # --->(720, 1280, 3)
    print(im.size)  # --->2764800
    im = im[0:1025, 0:3650, :]
    imsave(r"F:\ProjectSet\Shadow\MkTu\Imdc\ShadowMkTuImdc4.jpg", im, dpi=300)
    plt.imshow(im)
    plt.show()
    pass


if __name__ == "__main__":
    main()
