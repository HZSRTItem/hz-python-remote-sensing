# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : GDALFuncs.py
@Time    : 2023/10/29 10:22
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of GDALFuncs
-----------------------------------------------------------------------------"""
from SRTCodes.Utils import changext, changefiledirname


def buildvrtSeparateList(filelist, to_filelist=None, channels=None, to_dirname=None, options=None):
    if options is None:
        options = []

    for i, fn in enumerate(filelist):

        if channels is None:
            channel = 1
        else:
            channel = channels[i]

        if to_filelist is None:
            to_fn = changext(fn, "_{0}.vrt".format(channel))
        else:
            to_fn = to_filelist[i]

        if to_dirname is not None:
            to_fn = changefiledirname(to_fn, to_dirname)

        line = "gdalbuildvrt -separate -b " + str(channel) + " "
        for opt in options:
            line += opt + " "
        line += to_fn + " " + fn

        print(line)


def buildvrtSeparateList2(slist, to_dirname=None, options=None):
    filelist = []
    channels = []
    to_filelist = []

    for i, s0 in enumerate(slist):
        if len(s0) == 1:
            fn = s0[0]
            channel = 1
            to_fn = changext(fn, "_{0}.vrt".format(channel))
        elif len(s0) == 2:
            fn = s0[0]
            channel = s0[1]
            to_fn = changext(fn, "_{0}.vrt".format(channel))
        else:
            fn = s0[0]
            channel = s0[1]
            to_fn = s0[2]
        filelist.append(fn)
        channels.append(channel)
        to_filelist.append(to_fn)

    buildvrtSeparateList(filelist=filelist, to_filelist=to_filelist, channels=channels, to_dirname=to_dirname,
                         options=options)

def main():
    #
    buildvrtSeparateList2(
        slist=[
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 2, "Blue.vrt"],
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 3, "Green.vrt"],
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 4, "Red.vrt"],
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 8, "NIR.vrt"],
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 24, "AS_VV.vrt"],
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 25, "AS_VH.vrt"],
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 27, "DE_VV.vrt"],
            [r"F:\ProjectSet\Shadow\ChengDu\Image\cd_shadow_as_de_1.tif", 28, "DE_VH.vrt"],
            [r"H:\ChengDu\AS\C2\C11_resample.dat", 1, r"AS_C11.vrt"],
            [r"H:\ChengDu\AS\C2\C12_imag_resample.dat", 1, r"AS_C12_imag.vrt"],
            [r"H:\ChengDu\AS\C2\C12_real_resample.dat", 1, r"AS_C12_real.vrt"],
            [r"H:\ChengDu\AS\C2\C22_resample.dat", 1, r"AS_C22.vrt"],
            [r"H:\ChengDu\DE\C2\C11.tif", 1, r"DE_C11.vrt"],
            [r"H:\ChengDu\DE\C2\C12_imag.tif", 1, r"DE_C12_imag.vrt"],
            [r"H:\ChengDu\DE\C2\C12_real.tif", 1, r"DE_C12_real.vrt"],
            [r"H:\ChengDu\DE\C2\C22.tif", 1, r"DE_C22.vrt"],
        ],
        to_dirname=r"H:\ChengDu\1",
        options=[]
    )
    pass

if __name__ == "__main__":
    main()
