# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZY5MTestFunc.py
@Time    : 2023/9/5 10:31
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZY5MTestFunc
-----------------------------------------------------------------------------"""
import os
import tarfile

import numpy as np
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.NumpyUtils import minmaxData
from SRTCodes.Utils import changext, readJson


def main():
    dirname = r"K:\zhongdianyanfa\柬埔寨-采样区-2米-高分-资源卫星-L1A-无云"
    to_dirname = r"K:\zhongdianyanfa\柬埔寨-采样区-2米-高分-资源卫星-L1A-无云\Temp"
    for root, dirs, files in os.walk(dirname):
        for fn in files:
            ff = os.path.join(root, fn)
            print(ff)
            f = tarfile.open(ff, "r")
            members = []
            for name in f.getnames():
                to_fn = os.path.join(to_dirname, name)
                if os.path.splitext(to_fn)[1] == ".jpg":
                    if not os.path.isfile(to_fn):
                        members.append(f.getmember(name))
            if members:
                print(len(members))
                f.extractall(to_dirname, members=members)

    pass


def method_name4():
    d = readJson(r"F:\ProjectSet\Huo\shufen\Baoxing_updated_2.json")
    features = d["features"]
    d1 = {}
    for i, feat in enumerate(features):
        print("{0}\t{1}\t{2}".format(i + 1, feat["type"], feat["geometry"]["type"]))


def method_name3():
    files = [
        r"E:\Anaconda3\Scripts\gdal_calc.py"
        , r"E:\Anaconda3\Scripts\gdal_edit.py"
        , r"E:\Anaconda3\Scripts\gdal_fillnodata.py"
        , r"E:\Anaconda3\Scripts\gdal_merge.py"
        , r"E:\Anaconda3\Scripts\gdal_pansharpen.py"
        , r"E:\Anaconda3\Scripts\gdal_polygonize.py"
        , r"E:\Anaconda3\Scripts\gdal_proximity.py"
        , r"E:\Anaconda3\Scripts\gdal_retile.py"
        , r"E:\Anaconda3\Scripts\gdal_sieve.py"
        , r"E:\Anaconda3\Scripts\gdal2tiles.py"
        , r"E:\Anaconda3\Scripts\gdal2xyz.py"
        , r"E:\Anaconda3\Scripts\gdalattachpct.py"
        , r"E:\Anaconda3\Scripts\gdalcompare.py"
        , r"E:\Anaconda3\Scripts\gdalmove.py"
        , r"E:\Anaconda3\Scripts\ogr_layer_algebra.py"
        , r"E:\Anaconda3\Scripts\ogrmerge.py"
        , r"E:\Anaconda3\Scripts\pct2rgb.py"
        , r"E:\Anaconda3\Scripts\rgb2pct.py"]
    for f in files:
        fn = os.path.split(f)[1]
        to_fn = os.path.join(r"F:\code\bin", changext(fn, "_r.bat"))
        print(to_fn)
        with open(to_fn, "w") as fw:
            fw.write("@python {0} %*".format(f))
    # argv = ["f", "dsf", "dsfsd", "sdfsd", "-filelist", r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\ZY1F_VNIC_E105.2_N11.6_20221231_L1B0000322465\retiles1\filelist.txt", "dfsfagered", "dsfesdf"]
    # printList("argv", argv)
    # i = 0
    # while i < len(argv):
    #     if argv[i] == "-filelist":
    #         argv2 = argv.copy()
    #         fn = argv2[i + 1]
    #         argv2.pop(i)
    #         argv2.pop(i)
    #         with open(fn, "r", encoding="utf-8") as fr:
    #             for line in fr:
    #                 line = line.strip()
    #                 if os.path.isfile(line):
    #                     argv2.append(line)
    #         argv = argv2
    #     i+=1
    # printList("argv", argv)


def method_name2():
    # 把金边的影响转换为RGBN和Byte类型
    dirname = r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\ZY1F_VNIC_E105.2_N11.6_20221231_L1B0000322465\retiles1"
    for f in os.listdir(dirname):
        ff = os.path.join(dirname, f)
        if os.path.splitext(ff)[1] == ".tif":
            if os.path.isfile(ff):
                to_f = changext(ff, "_torgb255.dat")
                print(to_f)
                gr = GDALRaster(ff)
                d_blue = gr.readGDALBand(1)
                d_green = gr.readGDALBand(2)
                d_red = gr.readGDALBand(3)
                d_nir = gr.readGDALBand(4)
                d_blue = minmaxData(d_blue, 300, 3000, 0, 255)
                d_green = minmaxData(d_green, 300, 3000, 0, 255)
                d_red = minmaxData(d_red, 300, 3000, 0, 255)
                d_nir = minmaxData(d_nir, 300, 3000, 0, 255)
                d = np.concatenate([[d_blue], [d_green], [d_red], [d_nir]])
                gr.save(d.astype("int8"), to_f, descriptions=["Blue", "Green", "Red", "NIR"], dtype=gdal.GDT_Byte)


def method_name():
    gz_fn = r"K:\zhongdianyanfa\柬埔寨数据-第五批\优于5米2021-2022\CB04A_WPM_E105.4_N11.2_20221209_L1A0000393107.tar.gz"
    f = tarfile.open(gz_fn)
    print(f.getmember('CB04A_WPM_E105.4_N11.2_20221209_L1A0000393107.jpg').get_info())
    f.close()


if __name__ == "__main__":
    main()
