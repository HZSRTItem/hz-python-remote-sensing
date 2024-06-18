# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHTemp2.py
@Time    : 2024/6/5 15:56
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHTemp2
-----------------------------------------------------------------------------"""
import os

import pandas as pd

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import vrtAddDescriptions
from SRTCodes.Utils import FRW
from Shadow.ShadowData import ShadowData


def main():
    def func1():
        geo_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif"
        gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.tif")
        for i, name in enumerate(gr.names):
            print("gdalbuildvrt -r bilinear -b {} {} {}".format(
                i + 1,
                os.path.join(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1_VRTS", "{}.vrt".format(name))
                , geo_fn))

    def func2():
        sd = ShadowData()
        raster_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\cat\filelist2_vrt.vrt"  # 青岛
        # optics
        sd.addGDALData(raster_fn, "Blue")
        sd.addGDALData(raster_fn, "Green")
        sd.addGDALData(raster_fn, "Red")
        sd.addGDALData(raster_fn, "NIR")
        sd.extractNDVI("Red", "NIR", "NDVI")
        sd.extractNDWI("Green", "NIR", "NDWI")
        print("optics")
        # AS SAR
        sar_t = "AS"
        vv_n = sd.addGDALData(raster_fn, sar_t + "_VV")
        # sd.update10EDivide10(vv_n)
        vh_n = sd.addGDALData(raster_fn, sar_t + "_VH")
        # sd.update10EDivide10(vh_n)
        sd.extractSigmaRatio(sar_t + "_VV", sar_t + "_VH", sar_t + "_VHDVV")
        c11_n = sd.addGDALData(raster_fn, sar_t + "_C11")
        c12_imag_n = sd.addGDALData(raster_fn, sar_t + "_C12_imag")
        c12_real_n = sd.addGDALData(raster_fn, sar_t + "_C12_real")
        c22_n = sd.addGDALData(raster_fn, sar_t + "_C22")
        lamd1_n, lamd2_n = sd.extractC2EIG(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_Lambda1", sar_t + "_Lambda2")
        sd.extractSPAN(c11_n, c22_n, sar_t + "_SPAN")
        sd.extractEpsilon(c11_n, c22_n, sar_t + "_Epsilon")
        sd.extractMu(c12_real_n, c11_n, c22_n, sar_t + "_Mu")
        sd.extractRVI(c11_n, c22_n, sar_t + "_RVI")
        sd.extractDegreePolarization(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_m")
        sd.extractBeta(lamd1_n, lamd2_n, sar_t + "_Beta")
        print("AS SAR")
        # DE SAR
        sar_t = "DE"
        vv_n = sd.addGDALData(raster_fn, sar_t + "_VV")
        # sd.update10EDivide10(vv_n)
        vh_n = sd.addGDALData(raster_fn, sar_t + "_VH")
        # sd.update10EDivide10(vh_n)
        sd.extractSigmaRatio(sar_t + "_VV", sar_t + "_VH", sar_t + "_VHDVV")
        c11_n = sd.addGDALData(raster_fn, sar_t + "_C11")
        c12_imag_n = sd.addGDALData(raster_fn, sar_t + "_C12_imag")
        c12_real_n = sd.addGDALData(raster_fn, sar_t + "_C12_real")
        c22_n = sd.addGDALData(raster_fn, sar_t + "_C22")
        lamd1_n, lamd2_n = sd.extractC2EIG(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_Lambda1", sar_t + "_Lambda2")
        sd.extractSPAN(c11_n, c22_n, sar_t + "_SPAN")
        sd.extractEpsilon(c11_n, c22_n, sar_t + "_Epsilon")
        sd.extractMu(c12_real_n, c11_n, c22_n, sar_t + "_Mu")
        sd.extractRVI(c11_n, c22_n, sar_t + "_RVI")
        sd.extractDegreePolarization(c11_n, c22_n, c12_real_n, c12_imag_n, sar_t + "_m")
        sd.extractBeta(lamd1_n, lamd2_n, sar_t + "_Beta")
        print("DE SAR")
        sd.print()
        sd.saveToSingleImageFile(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\1", "SH2_QD1", raster_fn,
                                 vrt_fn="SH2_QD1.vrt")

    def func3():
        name_list = [
            "AS_angle", "AS_VH", "AS_VV", "Blue", "Green",
            "Red", "NIR", "B11", "B12", "DE_angle",
            "DE_VH", "DE_VV", "AS_C11", "AS_C12_imag", "AS_C12_real",
            "AS_C22", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22",
        ]
        vrtAddDescriptions(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\cat\filelist2_vrt.vrt",
                           r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\cat\filelist2_vrt.vrt",
                           name_list)

    def func4():
        df = pd.read_excel(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH青岛C2影像.xlsx", sheet_name="Sheet3")
        print(df)
        FRW(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\filelist2.txt").writeLines(df["FILE"].tolist())
        vrtAddDescriptions(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2.vrt",
                           r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\2\SHH2_QD2.vrt",
                           df["NAME"].tolist()
                           )

    func4()
    pass


if __name__ == "__main__":
    main()
