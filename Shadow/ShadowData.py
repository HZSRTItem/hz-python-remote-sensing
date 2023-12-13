# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowData.py
@Time    : 2023/9/12 19:24
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowData
-----------------------------------------------------------------------------"""
import os.path

import numpy as np
from osgeo import gdal

from SRTCodes.GDALRasterIO import GDALRaster, GDALRasterChannel
from SRTCodes.GDALUtils import SRTRasterConcat
from SRTCodes.Utils import readcsv, printList
from Shadow.ShadowFuncs import calEIG


class ShadowData(GDALRasterChannel):
    """ Shadow Data
    OPTICS
        1. $R, G, B, N$：原始波段
        2. $NDVI$：归一化植被指数 $\frac{N-R}{N+R}$
        3. $NDWI$：归一化水体指数 $\frac{G-R}{G+R}$


    SAR
        1. $\sigma_{VV}$：后向散射系数VV
        2. $\sigma_{VH}$：后向散射系数VH
        3. $C_{11}$：协方差矩阵主对角线第1个元素
        4. $C_{22}$：协方差矩阵主对角线第2个元素
        5. $SPAN$：总能量 $C_{11} + C_{22}$
        6. $\epsilon$：交叉极化比 $\frac{C_{11}}{C_{22}}$
        7. $\mu$：交叉极化相关系数 $\frac{C_{12}}{\sqrt{C_{11}*C_{22}}}$
        8. $RVI$：雷达植被指数 $\frac{4C_{22}}{SPAN}$
        9. lambda：特征值分解的结果，特征值量化了散射机制的主导地位
        10. m: 电磁波的极化状态以极化程度（0 ≤ m ≤ 1）来表征。极化程度定义为波的极化部分的（平均）强度与波的（平均）总强度之比
               对于大多数分布式目标，VV 和 HH 由一阶散射（即没有多重反射的直接反向散射）主导，而 HV（或 VH）是由于二阶和高阶散射
               （即涉及两个或多个散射体的两个或多个反射）。

    是不是可以在散射机制的主导型上分析
    """

    CSVS = {}

    def __init__(self):
        super(ShadowData, self).__init__()
        pass

    def addCSVData(self, csv_fn, field_name, csv_column_name=None):
        csv_fn = os.path.abspath(csv_fn)
        if csv_fn not in self.CSVS:
            self.CSVS[csv_fn] = readcsv(csv_fn)
        csv_d: dict = self.CSVS[csv_fn]
        if csv_column_name is None:
            csv_column_name = field_name
        self.data[field_name] = csv_d[csv_column_name]
        if self.data[field_name] is None:
            print("Warning: can not read data from {0}:{1}".format(field_name, csv_column_name))
        return field_name

    def extractNDVI(self, red_key="red", nir_key="nir", this_key="ndvi"):
        """ $NDVI$ = $\frac{N-R}{N+R}$ """
        return self.extractNormalizedDifference(nir_key, red_key, this_key=this_key)

    def extractNDWI(self, green_key="green", nir_key="nir", this_key="ndwi"):
        """ $NDWI$ = $\frac{G-R}{G+R}$ """
        return self.extractNormalizedDifference(green_key, nir_key, this_key=this_key)

    def extractNormalizedDifference(self, first_key, second_key, this_key):
        """ (first − second) / (first + second) """
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractNormalizedDifference.".format(this_key))
        d_first = self.data[first_key]
        d_second = self.data[second_key]
        self.data[this_key] = (d_first - d_second) / (d_first + d_second)
        return this_key

    def extractSigmaRatio(self, vv_key, vh_key, this_key="vv_d_vh"):
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractSigmaRatio.".format(this_key))
        self.data[this_key] = self.data[vh_key] / (self.data[vv_key] + 0.00001)

    def extractSPAN(self, c11_key="c11", c22_key="c22", this_key="SPAN"):
        """ $SPAN$ = $C_{11} + C_{22}$ """
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractSPAN.".format(this_key))
        self.data[this_key] = self.data[c11_key] + self.data[c22_key]
        return this_key

    def extractEpsilon(self, c11_key="c11", c22_key="c22", this_key="epsilon"):
        """ $epsilon$ = $\frac{C_{11}}{C_{22}}$ """
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractEpsilon.".format(this_key))
        self.data[this_key] = self.data[c11_key] / self.data[c22_key]
        return this_key

    def extractMu(self, c12_real_key="c12_real", c11_key="c11", c22_key="c22", this_key="mu"):
        """ $\mu$：交叉极化相关系数 $\frac{C_{12}}{\sqrt{C_{11}*C_{22}}}$ """
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractMu.".format(this_key))
        d_c12_real = self.data[c12_real_key]
        d_c11 = self.data[c11_key]
        d_c22 = self.data[c22_key]
        self.data[this_key] = d_c12_real / np.sqrt(d_c11 * d_c22)
        return this_key

    def extractRVI(self, c11_key="c11", c22_key="c22", this_key="RVI"):
        """ $RVI$：雷达植被指数 $\frac{4C_{22}}{SPAN}$ """
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractRVI.".format(this_key))
        d_c11 = self.data[c11_key]
        d_c22 = self.data[c22_key]
        self.data[this_key] = 4 * d_c22 / (d_c11 + d_c22)
        return this_key

    def extractC2EIG(self, c11_key="c11", c22_key="c22", c12_real_key="c12_real", c12_imag_key="c12_imag",
                     c2_lambda1_key="c2_lambda_1", c2_lambda2_key="c2_lambda_2"):
        if c2_lambda1_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractC2EIG.".format(c2_lambda1_key))
        if c2_lambda2_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractC2EIG.".format(c2_lambda2_key))
        d_c11 = self.data[c11_key]
        d_c22 = self.data[c22_key]
        d_c12_real = self.data[c12_real_key]
        d_c12_imag = self.data[c12_imag_key]
        lamd1, lamd2 = calEIG(d_c11, d_c22, d_c12_real, d_c12_imag)
        self.data[c2_lambda1_key] = lamd1
        self.data[c2_lambda2_key] = lamd2
        return c2_lambda1_key, c2_lambda2_key

    def extractDegreePolarization(self, c11_key="c11", c22_key="c22", c12_real_key="c12_real", c12_imag_key="c12_imag",
                                  this_key="c2_m"):
        """ The degree of polarization is defined as the ratio of the (average) intensity of the polarized
         portion of the wave to that of the (average) total intensity of the wave. """
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractC2EIG.".format(this_key))
        d_c11 = self.data[c11_key]
        d_c22 = self.data[c22_key]
        d_c12_real = self.data[c12_real_key]
        d_c12_imag = self.data[c12_imag_key]
        c2 = d_c11 * d_c22 - (d_c12_real * d_c12_real + d_c12_imag * d_c12_imag)
        self.data[this_key] = np.sqrt(1 - 4 * (c2 / ((d_c11 + d_c22) ** 2)))
        return this_key

    def extractBeta(self, c2_lambda1_key="c2_lambda_1", c2_lambda2_key="c2_lambda_2", this_key="beta"):
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extractC2EIG.".format(this_key))
        lamd1 = self.data[c2_lambda1_key]
        lamd2 = self.data[c2_lambda2_key]
        self.data[this_key] = lamd1 / (lamd1 + lamd2)
        return this_key

    def extFeatureData(self, func, this_key, *keys):
        if this_key in self.data:
            print("Warning: key:\"{0}\" have in data. Func:extFeatureData.".format(this_key))
        datas = []
        for k in keys:
            datas.append(self.data[k])
        self.data[this_key] = func(*datas)
        return this_key

    def __setitem__(self, key, value):
        self.addData(key, value)

    def addData(self, key, data):
        if key in self.data:
            print("Warning: key:\"{0}\" have in data.".format(key))
        self.data[key] = data

    def print(self):
        printList("Data Fields", list(self.data.keys()))

    def saveToSingleImageFile(self, dirname, front_str="", geo_fn=None, vrt_fn=None):
        gr = self._getGR(geo_fn)
        filelist = []
        for k in self.data:
            save_fn = os.path.join(dirname, front_str + k + ".dat")
            print(save_fn)
            filelist.append(save_fn)
            gr.save(self.data[k], save_fn, descriptions=[k])
        if vrt_fn is not None:
            vrt_fn = os.path.join(dirname, vrt_fn)
            print(vrt_fn)
            vrt: gdal.Dataset = gdal.BuildVRT(vrt_fn, filelist, separate=True)
            ks = list(self.data.keys())
            for i in range(len(filelist)):
                b: gdal.Band = vrt.GetRasterBand(i + 1)
                b.SetDescription(ks[i])

    def update10Log10(self, this_key):
        self.data[this_key] = 10 * np.log10(self.data[this_key])
        return this_key

    def update10EDivide10(self, this_key):
        self.data[this_key] = np.power(10, self.data[this_key] / 10)
        return this_key


def extShadowImageQD():
    sd = ShadowData()
    raster_fn = r"F:\ProjectSet\Shadow\QingDao\Image\Image1\shadow_qd_im1.dat"  # 青岛
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

    sd.saveToSingleImageFile(r"F:\ProjectSet\Shadow\QingDao\Image\1", "QD_SH1_", raster_fn, vrt_fn="QD_SH1.vrt")


def dataSARToDB():
    sar_field_names = ["AS_VV", "AS_VH", "AS_C11", "AS_C22",
                       "AS_Lambda1", "AS_Lambda2", "AS_SPAN", "AS_Epsilon", "AS_VHDVV",
                       "DE_VV", "DE_VH", "DE_C11", "DE_C22",
                       "DE_Lambda1", "DE_Lambda2", "DE_SPAN", "DE_Epsilon", "DE_VHDVV", ]
    sd = ShadowData()
    raster_fn = r"F:\ProjectSet\Shadow\ChengDu\Image\1\CD_SH1.vrt"
    sd.addGDALDatas(raster_fn)
    for field_name in sar_field_names:
        print(field_name)
        sd.update10Log10(field_name)
    sd.saveToSingleImageFile(r"F:\ProjectSet\Shadow\ChengDu\Image\2", "CD_SH2_", raster_fn, vrt_fn="CD_SH2.vrt")


def extShadowImageCD():
    sd = ShadowData()
    raster_fn = r"F:\ProjectSet\Shadow\ChengDu\Image\2\CD_SH2.vrt"  # 成都
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
    sd.update10EDivide10(vv_n)
    vh_n = sd.addGDALData(raster_fn, sar_t + "_VH")
    sd.update10EDivide10(vh_n)
    sd.extractSigmaRatio(sar_t + "_VV", sar_t + "_VH", sar_t + "_VHDVV")
    # sd.update10Log10(vv_n)
    # sd.update10Log10(vh_n)
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
    sd.update10EDivide10(vv_n)
    vh_n = sd.addGDALData(raster_fn, sar_t + "_VH")
    sd.update10EDivide10(vh_n)
    sd.extractSigmaRatio(sar_t + "_VV", sar_t + "_VH", sar_t + "_VHDVV")
    # sd.update10Log10(vv_n)
    # sd.update10Log10(vh_n)
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

    sd.saveToSingleImageFile(r"F:\ProjectSet\Shadow\ChengDu\Image\5", "CD_SH5_", raster_fn, vrt_fn="CD_SH5.vrt")
    # sd.saveToSingleImageFile(r"F:\ProjectSet\Shadow\BeiJing\Image\1", "BJ_SH1_", raster_fn, vrt_fn="BJ_SH1.vrt")


def extShadowImageBJ():
    sd = ShadowData()
    raster_fn = r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3.vrt"  # 成都
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
    # sd.update10Log10(vv_n)
    # sd.update10Log10(vh_n)
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
    # sd.update10Log10(vv_n)
    # sd.update10Log10(vh_n)
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

    sd.saveToSingleImageFile(r"F:\ProjectSet\Shadow\BeiJing\Image\5", "BJ_SH5_", raster_fn, vrt_fn="BJ_SH5.vrt")


def main():
    extShadowImageQD()
    pass


def qdImageConcat():
    # splitChannelsToImages(r"F:\ProjectSet\Shadow\QingDao\Image\Image1\shadow_qd_im1.dat")
    src = SRTRasterConcat(r"F:\ProjectSet\Shadow\Release\QingDaoImages",
                          r"F:\ProjectSet\Shadow\Release\QD_SH.vrt",
                          is_q=False)
    im_fn_o = r"F:\ProjectSet\Shadow\QingDao\Image\1\QD_SH1.vrt"
    im_fn_optics_glcm = r"F:\ProjectSet\Shadow\QingDao\Image\GLCM\QD1_s2glcm.tif"
    im_fn_sar_glcm = r"F:\ProjectSet\Shadow\QingDao\Image\GLCM\QD2_s1glcm.tif"
    # OPTICS
    src.add(fn=im_fn_o, c="Blue")
    src.add(fn=im_fn_o, c="Green")
    src.add(fn=im_fn_o, c="Red")
    src.add(fn=im_fn_o, c="NIR")
    src.add(fn=im_fn_o, c="NDVI")
    src.add(fn=im_fn_o, c="NDWI")
    # OPTICS GLCM
    src.add(fn=im_fn_optics_glcm, c="Gray_asm")
    src.add(fn=im_fn_optics_glcm, c="Gray_contrast")
    src.add(fn=im_fn_optics_glcm, c="Gray_corr")
    src.add(fn=im_fn_optics_glcm, c="Gray_var")
    src.add(fn=im_fn_optics_glcm, c="Gray_idm")
    src.add(fn=im_fn_optics_glcm, c="Gray_savg")
    src.add(fn=im_fn_optics_glcm, c="Gray_svar")
    src.add(fn=im_fn_optics_glcm, c="Gray_sent")
    src.add(fn=im_fn_optics_glcm, c="Gray_ent")
    src.add(fn=im_fn_optics_glcm, c="Gray_dvar")
    src.add(fn=im_fn_optics_glcm, c="Gray_dent")
    src.add(fn=im_fn_optics_glcm, c="Gray_imcorr1")
    src.add(fn=im_fn_optics_glcm, c="Gray_imcorr2")
    src.add(fn=im_fn_optics_glcm, c="Gray_maxcorr")
    src.add(fn=im_fn_optics_glcm, c="Gray_diss")
    src.add(fn=im_fn_optics_glcm, c="Gray_inertia")
    src.add(fn=im_fn_optics_glcm, c="Gray_shade")
    src.add(fn=im_fn_optics_glcm, c="Gray_prom")
    # AS
    src.add(fn=im_fn_o, c="AS_VV")
    src.add(fn=im_fn_o, c="AS_VH")
    src.add(fn=im_fn_o, c="AS_VHDVV")
    src.add(fn=im_fn_o, c="AS_C11")
    src.add(fn=im_fn_o, c="AS_C12_imag")
    src.add(fn=im_fn_o, c="AS_C12_real")
    src.add(fn=im_fn_o, c="AS_C22")
    src.add(fn=im_fn_o, c="AS_Lambda1")
    src.add(fn=im_fn_o, c="AS_Lambda2")
    src.add(fn=im_fn_o, c="AS_SPAN")
    src.add(fn=im_fn_o, c="AS_Epsilon")
    src.add(fn=im_fn_o, c="AS_Mu")
    src.add(fn=im_fn_o, c="AS_RVI")
    src.add(fn=im_fn_o, c="AS_m")
    src.add(fn=im_fn_o, c="AS_Beta")
    # AS GLCM
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_asm")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_contrast")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_corr")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_var")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_idm")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_savg")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_svar")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_sent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_ent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_dvar")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_dent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_imcorr1")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_imcorr2")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_maxcorr")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_diss")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_inertia")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_shade")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_prom")
    # DE
    src.add(fn=im_fn_o, c="DE_VV")
    src.add(fn=im_fn_o, c="DE_VH")
    src.add(fn=im_fn_o, c="DE_VHDVV")
    src.add(fn=im_fn_o, c="DE_C11")
    src.add(fn=im_fn_o, c="DE_C12_imag")
    src.add(fn=im_fn_o, c="DE_C12_real")
    src.add(fn=im_fn_o, c="DE_C22")
    src.add(fn=im_fn_o, c="DE_Lambda1")
    src.add(fn=im_fn_o, c="DE_Lambda2")
    src.add(fn=im_fn_o, c="DE_SPAN")
    src.add(fn=im_fn_o, c="DE_Epsilon")
    src.add(fn=im_fn_o, c="DE_Mu")
    src.add(fn=im_fn_o, c="DE_RVI")
    src.add(fn=im_fn_o, c="DE_m")
    src.add(fn=im_fn_o, c="DE_Beta")
    # DE GLCM
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_asm")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_contrast")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_corr")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_var")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_idm")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_savg")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_svar")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_sent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_ent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_dvar")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_dent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_imcorr1")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_imcorr2")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_maxcorr")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_diss")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_inertia")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_shade")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_prom")
    src.fit()


def bjImageConcat():
    # splitChannelsToImages(r"F:\ProjectSet\Shadow\QingDao\Image\Image1\shadow_qd_im1.dat")
    src = SRTRasterConcat(r"F:\ProjectSet\Shadow\Release\BeiJingImages",
                          r"F:\ProjectSet\Shadow\Release\BJ_SH.vrt",
                          is_q=False)
    im_fn_o = r"F:\ProjectSet\Shadow\BeiJing\Image\5\BJ_SH5.vrt"
    im_fn_optics_glcm = r"F:\ProjectSet\Shadow\BeiJing\Image\GLCM\BJ1_s2glcm.tif"
    im_fn_sar_glcm = r"F:\ProjectSet\Shadow\BeiJing\Image\GLCM\BJ2_s1glcm.tif"
    # OPTICS
    src.add(fn=im_fn_o, c="Blue")
    src.add(fn=im_fn_o, c="Green")
    src.add(fn=im_fn_o, c="Red")
    src.add(fn=im_fn_o, c="NIR")
    src.add(fn=im_fn_o, c="NDVI")
    src.add(fn=im_fn_o, c="NDWI")
    # OPTICS GLCM
    src.add(fn=im_fn_optics_glcm, c="Gray_asm")
    src.add(fn=im_fn_optics_glcm, c="Gray_contrast")
    src.add(fn=im_fn_optics_glcm, c="Gray_corr")
    src.add(fn=im_fn_optics_glcm, c="Gray_var")
    src.add(fn=im_fn_optics_glcm, c="Gray_idm")
    src.add(fn=im_fn_optics_glcm, c="Gray_savg")
    src.add(fn=im_fn_optics_glcm, c="Gray_svar")
    src.add(fn=im_fn_optics_glcm, c="Gray_sent")
    src.add(fn=im_fn_optics_glcm, c="Gray_ent")
    src.add(fn=im_fn_optics_glcm, c="Gray_dvar")
    src.add(fn=im_fn_optics_glcm, c="Gray_dent")
    src.add(fn=im_fn_optics_glcm, c="Gray_imcorr1")
    src.add(fn=im_fn_optics_glcm, c="Gray_imcorr2")
    src.add(fn=im_fn_optics_glcm, c="Gray_maxcorr")
    src.add(fn=im_fn_optics_glcm, c="Gray_diss")
    src.add(fn=im_fn_optics_glcm, c="Gray_inertia")
    src.add(fn=im_fn_optics_glcm, c="Gray_shade")
    src.add(fn=im_fn_optics_glcm, c="Gray_prom")
    # AS
    src.add(fn=im_fn_o, c="AS_VV")
    src.add(fn=im_fn_o, c="AS_VH")
    src.add(fn=im_fn_o, c="AS_VHDVV")
    src.add(fn=im_fn_o, c="AS_C11")
    src.add(fn=im_fn_o, c="AS_C12_imag")
    src.add(fn=im_fn_o, c="AS_C12_real")
    src.add(fn=im_fn_o, c="AS_C22")
    src.add(fn=im_fn_o, c="AS_Lambda1")
    src.add(fn=im_fn_o, c="AS_Lambda2")
    src.add(fn=im_fn_o, c="AS_SPAN")
    src.add(fn=im_fn_o, c="AS_Epsilon")
    src.add(fn=im_fn_o, c="AS_Mu")
    src.add(fn=im_fn_o, c="AS_RVI")
    src.add(fn=im_fn_o, c="AS_m")
    src.add(fn=im_fn_o, c="AS_Beta")
    # AS GLCM
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_asm")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_contrast")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_corr")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_var")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_idm")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_savg")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_svar")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_sent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_ent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_dvar")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_dent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_imcorr1")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_imcorr2")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_maxcorr")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_diss")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_inertia")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_shade")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_prom")
    # DE
    src.add(fn=im_fn_o, c="DE_VV")
    src.add(fn=im_fn_o, c="DE_VH")
    src.add(fn=im_fn_o, c="DE_VHDVV")
    src.add(fn=im_fn_o, c="DE_C11")
    src.add(fn=im_fn_o, c="DE_C12_imag")
    src.add(fn=im_fn_o, c="DE_C12_real")
    src.add(fn=im_fn_o, c="DE_C22")
    src.add(fn=im_fn_o, c="DE_Lambda1")
    src.add(fn=im_fn_o, c="DE_Lambda2")
    src.add(fn=im_fn_o, c="DE_SPAN")
    src.add(fn=im_fn_o, c="DE_Epsilon")
    src.add(fn=im_fn_o, c="DE_Mu")
    src.add(fn=im_fn_o, c="DE_RVI")
    src.add(fn=im_fn_o, c="DE_m")
    src.add(fn=im_fn_o, c="DE_Beta")
    # DE GLCM
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_asm")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_contrast")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_corr")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_var")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_idm")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_savg")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_svar")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_sent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_ent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_dvar")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_dent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_imcorr1")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_imcorr2")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_maxcorr")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_diss")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_inertia")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_shade")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_prom")
    src.fit()


def cdImageConcat():
    # splitChannelsToImages(r"F:\ProjectSet\Shadow\QingDao\Image\Image1\shadow_qd_im1.dat")
    src = SRTRasterConcat(r"F:\ProjectSet\Shadow\Release\ChengDuImages",
                          r"F:\ProjectSet\Shadow\Release\CD_SH.vrt",
                          is_q=False)
    im_fn_o = r"F:\ProjectSet\Shadow\ChengDu\Image\1\CD_SH1.vrt"
    im_fn_optics_glcm = r"F:\ProjectSet\Shadow\ChengDu\Image\GLCM\CD1_s2glcm.tif"
    im_fn_sar_glcm = r"F:\ProjectSet\Shadow\ChengDu\Image\GLCM\CD2_s1glcm.tif"
    # OPTICS
    src.add(fn=im_fn_o, c="Blue")
    src.add(fn=im_fn_o, c="Green")
    src.add(fn=im_fn_o, c="Red")
    src.add(fn=im_fn_o, c="NIR")
    src.add(fn=im_fn_o, c="NDVI")
    src.add(fn=im_fn_o, c="NDWI")
    # OPTICS GLCM
    src.add(fn=im_fn_optics_glcm, c="Gray_asm")
    src.add(fn=im_fn_optics_glcm, c="Gray_contrast")
    src.add(fn=im_fn_optics_glcm, c="Gray_corr")
    src.add(fn=im_fn_optics_glcm, c="Gray_var")
    src.add(fn=im_fn_optics_glcm, c="Gray_idm")
    src.add(fn=im_fn_optics_glcm, c="Gray_savg")
    src.add(fn=im_fn_optics_glcm, c="Gray_svar")
    src.add(fn=im_fn_optics_glcm, c="Gray_sent")
    src.add(fn=im_fn_optics_glcm, c="Gray_ent")
    src.add(fn=im_fn_optics_glcm, c="Gray_dvar")
    src.add(fn=im_fn_optics_glcm, c="Gray_dent")
    src.add(fn=im_fn_optics_glcm, c="Gray_imcorr1")
    src.add(fn=im_fn_optics_glcm, c="Gray_imcorr2")
    src.add(fn=im_fn_optics_glcm, c="Gray_maxcorr")
    src.add(fn=im_fn_optics_glcm, c="Gray_diss")
    src.add(fn=im_fn_optics_glcm, c="Gray_inertia")
    src.add(fn=im_fn_optics_glcm, c="Gray_shade")
    src.add(fn=im_fn_optics_glcm, c="Gray_prom")
    # AS
    src.add(fn=im_fn_o, c="AS_VV")
    src.add(fn=im_fn_o, c="AS_VH")
    src.add(fn=im_fn_o, c="AS_VHDVV")
    src.add(fn=im_fn_o, c="AS_C11")
    src.add(fn=im_fn_o, c="AS_C12_imag")
    src.add(fn=im_fn_o, c="AS_C12_real")
    src.add(fn=im_fn_o, c="AS_C22")
    src.add(fn=im_fn_o, c="AS_Lambda1")
    src.add(fn=im_fn_o, c="AS_Lambda2")
    src.add(fn=im_fn_o, c="AS_SPAN")
    src.add(fn=im_fn_o, c="AS_Epsilon")
    src.add(fn=im_fn_o, c="AS_Mu")
    src.add(fn=im_fn_o, c="AS_RVI")
    src.add(fn=im_fn_o, c="AS_m")
    src.add(fn=im_fn_o, c="AS_Beta")
    # AS GLCM
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_asm")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_contrast")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_corr")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_var")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_idm")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_savg")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_svar")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_sent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_ent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_dvar")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_dent")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_imcorr1")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_imcorr2")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_maxcorr")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_diss")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_inertia")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_shade")
    src.add(fn=im_fn_sar_glcm, c="AS_GLCM_prom")
    # DE
    src.add(fn=im_fn_o, c="DE_VV")
    src.add(fn=im_fn_o, c="DE_VH")
    src.add(fn=im_fn_o, c="DE_VHDVV")
    src.add(fn=im_fn_o, c="DE_C11")
    src.add(fn=im_fn_o, c="DE_C12_imag")
    src.add(fn=im_fn_o, c="DE_C12_real")
    src.add(fn=im_fn_o, c="DE_C22")
    src.add(fn=im_fn_o, c="DE_Lambda1")
    src.add(fn=im_fn_o, c="DE_Lambda2")
    src.add(fn=im_fn_o, c="DE_SPAN")
    src.add(fn=im_fn_o, c="DE_Epsilon")
    src.add(fn=im_fn_o, c="DE_Mu")
    src.add(fn=im_fn_o, c="DE_RVI")
    src.add(fn=im_fn_o, c="DE_m")
    src.add(fn=im_fn_o, c="DE_Beta")
    # DE GLCM
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_asm")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_contrast")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_corr")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_var")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_idm")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_savg")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_svar")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_sent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_ent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_dvar")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_dent")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_imcorr1")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_imcorr2")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_maxcorr")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_diss")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_inertia")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_shade")
    src.add(fn=im_fn_sar_glcm, c="DE_GLCM_prom")
    src.fit()


def method_name():
    dirname = r"F:\ProjectSet\Shadow\QingDao\Image\Image1\2"
    for f in os.listdir(dirname):
        ff = os.path.join(dirname, f)
        if os.path.isfile(ff):
            if os.path.splitext(ff)[1] == ".dat":
                gr = GDALRaster(ff)
                print(gr.n_rows, gr.n_columns, gr.n_channels)


if __name__ == "__main__":
    extShadowImageCD()
