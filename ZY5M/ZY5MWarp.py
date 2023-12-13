# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ZY5MWarp.py
@Time    : 2023/9/5 17:19
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ZY5MWarp
gdal.Warp(destNameOrDestDS, srcDSOrSrcDSTab, **kwargs)


-----------------------------------------------------------------------------"""
import os.path
import tarfile

import pandas as pd

from SRTCodes.GDALRasterIO import GDALRasterWarp
from SRTCodes.Utils import findfile, changext, readJson


def wktToCoors(wkt: str):
    wkt = str(wkt)
    wkt = wkt.strip()
    wkt_coor = wkt[9:-2]
    wkt_coors = wkt_coor.split(",")
    coors = []
    for coor_str in wkt_coors:
        tmp = coor_str.split(' ')
        coors.append([float(tmp[0]), float(tmp[1])])
    return coors


class ZY5MGDALRasterWarp:
    TARS = {}

    def __init__(self):
        super().__init__()
        self.jpz_dirname = r"K:\zhongdianyanfa\jpz_5"
        self.jpz_tmp_dirname = r"K:\zhongdianyanfa\jpz_5\ChuLi"

    def transJpgGEO(self, csv_fn):
        df = pd.read_csv(csv_fn)
        fns = df["datafile"].values.tolist()
        wkts = df["data_wkt"].values.tolist()
        to_dirname = self.jpz_tmp_dirname

        for i in range(len(fns)):
            print("> {0}: {1}".format(i + 1, fns[i]))
            fn = self.findFile(fns[i])
            if fn == "":
                continue
            coors = wktToCoors(wkts[i])
            jpg_fns = self.extTarJpg(fn, to_dirname)
            for jpg_fn in jpg_fns:
                to_fn = changext(jpg_fn, "_geo.tif")
                print("  + {0}".format(to_fn))
                if os.path.isfile(to_fn):
                    continue
                self.jpgToGeoTiff(coors, jpg_fn, to_fn)

    def jpgToGeoTiff(self, coors, jpg_fn, to_fn):
        zy5m_grw = GDALRasterWarp()
        zy5m_grw.initGDALRaster(jpg_fn)
        zy5m_grw.addGCP(coors[0][0], coors[0][1], 0, 0, 0)
        zy5m_grw.addGCP(coors[1][0], coors[1][1], 0, zy5m_grw.n_columns, 0)
        zy5m_grw.addGCP(coors[2][0], coors[2][1], 0, zy5m_grw.n_columns, zy5m_grw.n_rows)
        zy5m_grw.addGCP(coors[3][0], coors[3][1], 0, 0, zy5m_grw.n_rows)
        zy5m_grw.warp(to_fn, xres=zy5m_grw.RESOLUTION_ANGLE * 3, yres=zy5m_grw.RESOLUTION_ANGLE * 3,
                      dtype="int8")

    @classmethod
    def extTarJpg(cls, fn, to_dirname):
        fns = []
        fn = os.path.abspath(fn)
        if fn not in cls.TARS:
            cls.TARS[fn] = tarfile.open(fn)
        tars = cls.TARS[fn]
        for tar in tars.getmembers():
            if os.path.splitext(tar.name)[1] == ".jpg":
                to_fn0 = os.path.join(to_dirname, tar.name)
                if not os.path.isfile(to_fn0):
                    tars.extract(tar, to_dirname)
                fns.append(to_fn0)
        return fns

    def findFile(self, fn):
        fn = str(fn)
        if fn.startswith("/"):
            fn = fn[1:]
        fn = findfile(self.jpz_dirname, fn)
        return fn

    def transJpgGEO2(self, dirname):
        for f in os.listdir(dirname):
            ff = os.path.join(dirname, f)
            if os.path.isfile(ff):
                if os.path.splitext(ff)[1] == ".json":
                    d1 = readJson(ff)
                    print(ff)
                    try:
                        coors = wktToCoors(d1["geom"])
                        jpg_fn = changext(ff, ".jpg")
                        to_fn = changext(ff, "_geo.tif")
                        self.jpgToGeoTiff(coors, jpg_fn, to_fn)
                    except:
                        print("Error")

    def warpImage(self, coor_csv_fn, raster_fn, to_fn):
        zy5m_grw = GDALRasterWarp()
        zy5m_grw.initGDALRaster(raster_fn)
        df = pd.read_csv(coor_csv_fn)
        for i in range(0, len(df), 2):
            zy5m_grw.addGCPImageGround(
                x_image=df["X"][i],
                y_image=df["Y"][i],
                x_ground=df["X"][i + 1],
                y_ground=df["Y"][i + 1])
        print("Start")
        zy5m_grw.warp(to_fn, dtype="uint16")


def main():
    # fn = r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\GF1_PMS1_E103.3_N10.4_20180212_L1A0002998578\GF1_PMS1_E103.3_N10.4_20180212_L1A0002998578-MSS1.jpg"
    # zy5m_grw = GDALRasterWarp()
    # zy5m_grw.initGDALRaster(fn)
    # # 103.185, 10.6028,103.527, 10.5308,103.453, 10.1891,103.111, 10.2611,103.185, 10.6028
    # zy5m_grw.addGCP(103.185, 10.6028, 0, 0, 0)
    # zy5m_grw.addGCP(103.527, 10.5308, 0, zy5m_grw.n_columns, 0)
    # zy5m_grw.addGCP(103.453, 10.1891, 0, zy5m_grw.n_columns, zy5m_grw.n_rows)
    # zy5m_grw.addGCP(103.111, 10.2611, 0, 0, zy5m_grw.n_rows)
    # zy5m_grw.warp(r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\GF1_PMS1_E103.3_N10.4_20180212_L1A0002998578\test1.tif",
    #               xres=zy5m_grw.RESOLUTION_ANGLE * 3, yres=zy5m_grw.RESOLUTION_ANGLE * 3, dtype="int8")
    zy5m_grw = ZY5MGDALRasterWarp()
    # zy5m_grw.transJpgGEO(r"K:\zhongdianyanfa\jpz_5\元数据优于5米\2017to2018\data_rectangle\data_rectangle.csv")
    # zy5m_grw.transJpgGEO2(r"K:\zhongdianyanfa\jpz_5\元数据优于5米\2021to2022\metadata")

    # zy5m_grw.warpImage(
    #     r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\ZY1F_VNIC_E105.2_N11.6_20221231_L1B0000322465\samples\spl1.csv",
    #     r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\ZY1F_VNIC_E105.2_N11.6_20221231_L1B0000322465\ZY1F_JinBian_1.tif",
    #     r"K:\zhongdianyanfa\jpz_5\good5m2021-2022\ZY1F_VNIC_E105.2_N11.6_20221231_L1B0000322465\ZY1F_JinBian_2.tif"
    # )

    zy5m_grw.warpImage(
        r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\ZY302_PMS_E105.0_N11.6_20180802_L1A0000356944\samples\spl1.csv",
        r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\ZY302_PMS_E105.0_N11.6_20180802_L1A0000356944\ZY302_JinBin17_1.tif",
        r"K:\zhongdianyanfa\jpz_5\good5m2017-2018\ZY302_PMS_E105.0_N11.6_20180802_L1A0000356944\ZY302_JinBin17_2.tif"
    )

    pass


if __name__ == "__main__":
    main()
