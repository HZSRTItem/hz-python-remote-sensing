# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowMain.py
@Time    : 2023/7/8 10:31
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ShadowMain
-----------------------------------------------------------------------------"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from SRTCodes.GDALRasterClassification import GDALRasterClassificationAccuracy
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import samplingToCSV
from SRTCodes.SRTSample import CSVSamples
from SRTCodes.Utils import DirFileName
from Shadow.ShadowDraw import cal_10log10, SampleDrawBox, SampleDrawScatter, ShadowSampleNumberDraw
from Shadow.ShadowImdC import ShadowImageClassification
from Shadow.ShadowSample import ShadowSampleNumber
from Shadow.ShadowTraining import ShadowCategoryTraining, trainRF, trainSvm

SAMPLE_FILENAME = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_train_test_spl_s1.csv"
FIG_DIR = r"F:\ProjectSet\Shadow\QingDao\mktu"


def drawShadowBox():
    csv_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_spl3_s1.csv"
    csv_spl = CSVSamples(csv_fn)

    csv_spl.fieldNameCategory("CNAME")
    csv_spl.fieldNameTag("TAG")
    csv_spl.addCategoryNames(["IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"])
    csv_spl.readData()
    csv_spl.featureCallBack("AS_VV", cal_10log10)
    csv_spl.featureCallBack("AS_VH", cal_10log10)
    csv_spl.featureCallBack("AS_C11", cal_10log10)
    csv_spl.featureCallBack("AS_C12_imag", cal_10log10)
    csv_spl.featureCallBack("AS_C12_real", cal_10log10)
    csv_spl.featureCallBack("AS_C22", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda1", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda2", cal_10log10)
    csv_spl.featureCallBack("DE_VV", cal_10log10)
    csv_spl.featureCallBack("DE_VH", cal_10log10)
    csv_spl.featureCallBack("DE_C11", cal_10log10)
    csv_spl.featureCallBack("DE_C12_imag", cal_10log10)
    csv_spl.featureCallBack("DE_C12_real", cal_10log10)
    csv_spl.featureCallBack("DE_C22", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda2", cal_10log10)

    spl_draw_box = SampleDrawBox()
    spl_draw_box.setSample(csv_spl)

    spl_draw_box.colors("red", "darkred", "lightgreen", "darkgreen", "yellow", "y", "lightblue", "darkblue")

    # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
    # "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"
    # "IS", "VEG", "SOIL", "WAT",
    spl_draw_box.addCategorys("IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH", )

    # "X", "Y", "CATEGORY", "SRT", "TEST", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
    # "AS_VV", "AS_VH", "AS_C11","AS_C22", "AS_Lambda1", "AS_Lambda2",
    #  "AS_C12_imag", "AS_C12_real",
    # "DE_VV", "DE_VH", "DE_C11",  "DE_C22", "DE_Lambda1", "DE_Lambda2",
    # "DE_C12_imag", "DE_C12_real",
    # "angle_AS", "angle_DE",
    spl_draw_box.addFeatures("AS_C11", "AS_C22")

    # "STATIFY_NOSH", "SEA", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH",
    spl_draw_box.addTags("STATIFY_NOSH", "SEA", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH")

    spl_draw_box.print()

    spl_draw_box.fit()

    # plt.ylim([-1, 1])
    # plt.savefig()
    plt.show()


def drawShadowScatter():
    csv_fn = SAMPLE_FILENAME
    csv_spl = CSVSamples(csv_fn)

    csv_spl.fieldNameCategory("CNAME")
    csv_spl.fieldNameTag("TAG")
    csv_spl.addCategoryNames(["IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"])
    csv_spl.readData()

    csv_spl.featureCallBack("AS_VV", cal_10log10)
    csv_spl.featureCallBack("AS_VH", cal_10log10)
    csv_spl.featureCallBack("AS_C11", cal_10log10)
    csv_spl.featureCallBack("AS_C22", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda1", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda2", cal_10log10)
    csv_spl.featureCallBack("DE_VV", cal_10log10)
    csv_spl.featureCallBack("DE_VH", cal_10log10)
    csv_spl.featureCallBack("DE_C11", cal_10log10)
    csv_spl.featureCallBack("DE_C22", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda2", cal_10log10)

    spl_draw_scatter = SampleDrawScatter()
    spl_draw_scatter.setSample(csv_spl)
    spl_draw_scatter.colors("red", "darkred", "lightgreen", "darkgreen", "yellow", "y", "lightblue", "darkblue")

    # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
    # "IS",    "VEG",    "SOIL",    "WAT",
    # "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
    spl_draw_scatter.addCategorys("IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH")

    # "X", "Y", "CATEGORY", "SRT", "TEST", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
    # "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
    # "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
    # "angle_AS", "angle_DE",
    spl_draw_scatter.addFeatures("AS_Lambda2", "DE_Lambda2")

    # "STATIFY_NOSH", "SEA", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH",
    spl_draw_scatter.addTags("STATIFY_NOSH", "SEA", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH")

    spl_draw_scatter.print()

    spl_draw_scatter.fit()

    # plt.ylim([0, 5000])
    # plt.xlim([0, 5000])
    plt.ylim([-50, 50])
    plt.xlim([-50, 50])
    plt.show()


def shadowTraining():
    mod_dir = r"F:\ProjectSet\Shadow\QingDao\Mods"
    spl_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_train_test_spl_s1.csv"

    csv_spl = CSVSamples(spl_fn)
    csv_spl.fieldNameCategory("CNAME")
    csv_spl.fieldNameTag("TAG")
    csv_spl.addCategoryNames(["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
    csv_spl.readData()
    csv_spl.featureCallBack("AS_VV", cal_10log10)
    csv_spl.featureCallBack("AS_VH", cal_10log10)
    csv_spl.featureCallBack("AS_C11", cal_10log10)
    csv_spl.featureCallBack("AS_C22", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda1", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda2", cal_10log10)
    csv_spl.featureCallBack("DE_VV", cal_10log10)
    csv_spl.featureCallBack("DE_VH", cal_10log10)
    csv_spl.featureCallBack("DE_C11", cal_10log10)
    csv_spl.featureCallBack("DE_C22", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda2", cal_10log10)
    csv_spl.is_trans = True
    csv_spl.is_scale_01 = True
    csv_spl.featureScaleMinMax("Blue", 299.76996, 2397.184)
    csv_spl.featureScaleMinMax("Green", 345.83414, 2395.735)
    csv_spl.featureScaleMinMax("Red", 177.79654, 2726.7026)
    csv_spl.featureScaleMinMax("NIR", 87.66086, 3498.4321)
    csv_spl.featureScaleMinMax("NDVI", -0.6, 0.9)
    csv_spl.featureScaleMinMax("NDWI", -0.7, 0.8)
    csv_spl.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
    csv_spl.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
    csv_spl.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
    csv_spl.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
    csv_spl.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
    csv_spl.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
    csv_spl.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
    csv_spl.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
    csv_spl.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
    csv_spl.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
    csv_spl.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
    csv_spl.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)

    sct = ShadowCategoryTraining(mod_dir, n_category=4, category_names=["IS", "VEG", "SOIL", "WAT"])
    sct.setSample(csv_spl)
    sct.addFrontFeatType("OPTICS", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI")
    sct.addModelType("RF", trainRF)
    sct.addModelType("SVM", trainSvm)
    # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
    sct.addSampleType("SPL_NOSH", "IS", "VEG", "SOIL", "WAT")
    sct.addSampleType("SPL_SH", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH")
    # "X", "Y", "CATEGORY", "SRT", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
    # "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
    # "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
    # "angle_AS", "angle_DE",
    sct.addFeatureType("AS_SIGMA", "AS_VV", "AS_VH")
    sct.addFeatureType("AS_C2", "AS_C11", "AS_C22")
    sct.addFeatureType("AS_LAMD", "AS_Lambda1", "AS_Lambda2")
    sct.addFeatureType("DE_SIGMA", "DE_VV", "DE_VH")
    sct.addFeatureType("DE_C2", "DE_C11", "DE_C22")
    sct.addFeatureType("DE_LAMD", "DE_Lambda1", "DE_Lambda2")
    # "STATIFY_NOSH", "SEA", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH",
    sct.addTagType("TAG", "STATIFY_NOSH", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "NIR_SH")
    sct.print()

    sct.train()

    pass


def shadowImdC():
    im_fn = r"G:\ImageData\QingDao\20211023\qd20211023\Temp\qd_im.dat"
    mod_dir = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910"

    sic = ShadowImageClassification(im_fn, mod_dir)

    sic.featureCallBack("AS_VV", cal_10log10)
    sic.featureCallBack("AS_VH", cal_10log10)
    sic.featureCallBack("AS_C11", cal_10log10)
    sic.featureCallBack("AS_C22", cal_10log10)
    sic.featureCallBack("AS_Lambda1", cal_10log10)
    sic.featureCallBack("AS_Lambda2", cal_10log10)
    sic.featureCallBack("DE_VV", cal_10log10)
    sic.featureCallBack("DE_VH", cal_10log10)
    sic.featureCallBack("DE_C11", cal_10log10)
    sic.featureCallBack("DE_C22", cal_10log10)
    sic.featureCallBack("DE_Lambda1", cal_10log10)
    sic.featureCallBack("DE_Lambda2", cal_10log10)

    sic.is_trans = True
    sic.is_scale_01 = True

    sic.featureScaleMinMax("Blue", 299.76996, 2397.184)
    sic.featureScaleMinMax("Green", 345.83414, 2395.735)
    sic.featureScaleMinMax("Red", 177.79654, 2726.7026)
    sic.featureScaleMinMax("NIR", 87.66086, 3498.4321)
    sic.featureScaleMinMax("NDVI", -0.6, 0.9)
    sic.featureScaleMinMax("NDWI", -0.7, 0.8)
    sic.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
    sic.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
    sic.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
    sic.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
    sic.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
    sic.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
    sic.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
    sic.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
    sic.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
    sic.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
    sic.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
    sic.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)

    sic.addCategory("IS", (255, 0, 0))
    sic.addCategory("VEG", (0, 255, 0))
    sic.addCategory("SOIL", (255, 255, 0))
    sic.addCategory("WAT", (0, 0, 255))

    sic.run()
    sic.print()
    pass


def trainSampleNumber():
    mod_dir = r"F:\ProjectSet\Shadow\QingDao\SampleNumber\Mods"
    spl_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_train_test_spl_s1.csv"

    csv_spl = CSVSamples(spl_fn)
    csv_spl.fieldNameCategory("CNAME")
    csv_spl.fieldNameTag("TAG")
    csv_spl.addCategoryNames(["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
    csv_spl.readData()
    csv_spl.featureCallBack("AS_VV", cal_10log10)
    csv_spl.featureCallBack("AS_VH", cal_10log10)
    csv_spl.featureCallBack("AS_C11", cal_10log10)
    csv_spl.featureCallBack("AS_C22", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda1", cal_10log10)
    csv_spl.featureCallBack("AS_Lambda2", cal_10log10)
    csv_spl.featureCallBack("DE_VV", cal_10log10)
    csv_spl.featureCallBack("DE_VH", cal_10log10)
    csv_spl.featureCallBack("DE_C11", cal_10log10)
    csv_spl.featureCallBack("DE_C22", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
    csv_spl.featureCallBack("DE_Lambda2", cal_10log10)
    csv_spl.is_trans = True
    csv_spl.is_scale_01 = True
    csv_spl.featureScaleMinMax("Blue", 299.76996, 2397.184)
    csv_spl.featureScaleMinMax("Green", 345.83414, 2395.735)
    csv_spl.featureScaleMinMax("Red", 177.79654, 2726.7026)
    csv_spl.featureScaleMinMax("NIR", 87.66086, 3498.4321)
    csv_spl.featureScaleMinMax("NDVI", -0.6, 0.9)
    csv_spl.featureScaleMinMax("NDWI", -0.7, 0.8)
    csv_spl.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
    csv_spl.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
    csv_spl.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
    csv_spl.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
    csv_spl.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
    csv_spl.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
    csv_spl.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
    csv_spl.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
    csv_spl.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
    csv_spl.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
    csv_spl.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
    csv_spl.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)

    ssn = ShadowSampleNumber(mod_dir, n_category=4, category_names=["IS", "VEG", "SOIL", "WAT"])
    ssn.setSample(csv_spl)
    ssn.addModelType("RF", trainRF)
    ssn.addModelType("SVM", trainSvm)
    ssn.sampleNumbers(300, 50, 20)
    # "X", "Y", "CATEGORY", "SRT", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
    # "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
    # "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
    # "angle_AS", "angle_DE",
    ssn.addFeatureType("OPTICS", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI")  # optic
    ssn.addFeatureType("OPTICS_AS", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",  # optic
                       "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2")  # AS
    ssn.addFeatureType("OPTICS_DE", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",  # optic
                       "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2")  # DE
    ssn.addFeatureType("OPTICS_AS_DE", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",  # optic
                       "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2",  # AS
                       "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2")  # DE
    ssn.sampleRandom()
    ssn.print()

    ssn.train()


def showSampleNumber():
    csv_fn = r"F:\ProjectSet\Shadow\QingDao\SampleNumber\Mods\20230909H184625\train_save_20230909H184625.csv"
    ssnd = ShadowSampleNumberDraw(csv_fn)
    ssnd.print()
    """
    * MOD_TYPE: 
      "RF", "SVM", 
    * SPL_TYPE: 
      "NO_SH", "SH", 
    * FEAT_TYPE: 
      "OPTICS", "OPTICS_AS", "OPTICS_AS_DE", "OPTICS_DE", 
    * SPL_NUMBER: 
      "500", "600", "700", "800", "900", "1000", 
    * COL_NAMES: 
      "OATrain", "KappaTrain", "IS UATrain", "IS PATrain", 
      "VEG UATrain", "VEG PATrain", "SOIL UATrain", "SOIL PATrain", 
      "WAT UATrain", "WAT PATrain", "OATest", "KappaTest", "IS UATest", 
      "IS PATest", "VEG UATest", "VEG PATest", "SOIL UATest", 
      "SOIL PATest", "WAT UATest", "WAT PATest", 
    """
    IS_sh_dict = {"color": 'r', "linestyle": '--', "marker": 'o', "markerfacecolor": 'r', "markersize": 6}
    IS_nosh_dict = {"color": 'r', "linestyle": '-', "marker": 'o', "markerfacecolor": 'r', "markersize": 6}
    VEG_sh_dict = {"color": 'g', "linestyle": '--', "marker": 'o', "markerfacecolor": 'g', "markersize": 6}
    VEG_nosh_dict = {"color": 'g', "linestyle": '-', "marker": 'o', "markerfacecolor": 'g', "markersize": 6}
    SOIL_sh_dict = {"color": 'y', "linestyle": '--', "marker": 'o', "markerfacecolor": 'y', "markersize": 6}
    SOIL_nosh_dict = {"color": 'y', "linestyle": '-', "marker": 'o', "markerfacecolor": 'y', "markersize": 6}
    WAT_sh_dict = {"color": 'b', "linestyle": '--', "marker": 'o', "markerfacecolor": 'b', "markersize": 6}
    WAT_nosh_dict = {"color": 'b', "linestyle": '-', "marker": 'o', "markerfacecolor": 'b', "markersize": 6}

    plt.subplots_adjust(
        # top=0.969, bottom=0.081, left=0.04, right=0.706, hspace=0.2, wspace=0.2
    )
    # ssnd.plot(mod_type="RF", spl_type="SH", feat_type="OPTICS", column_name="OATest")
    # ssnd.plot(mod_type="RF", spl_type="NO_SH", feat_type="OPTICS", column_name="OATest")
    ssnd.plot(mod_type="SVM", spl_type="SH", feat_type="OPTICS", column_name="OATest", name="OPTICS SHADOW",
              **IS_sh_dict)
    ssnd.plot(mod_type="SVM", spl_type="NO_SH", feat_type="OPTICS", column_name="OATest", name="OPTICS NO SHADOW",
              **IS_nosh_dict)
    # ssnd.plot(mod_type="SVM", spl_type="SH", feat_type="OPTICS_AS", column_name="OATest")
    # ssnd.plot(mod_type="SVM", spl_type="NO_SH", feat_type="OPTICS_AS", column_name="OATest")
    # ssnd.plot(mod_type="SVM", spl_type="SH", feat_type="OPTICS_DE", column_name="OATest")
    # ssnd.plot(mod_type="SVM", spl_type="NO_SH", feat_type="OPTICS_DE", column_name="OATest")
    ssnd.plot(mod_type="SVM", spl_type="SH", feat_type="OPTICS_AS_DE", column_name="OATest", name="OPTICS_AS_DE SHADOW",
              **VEG_sh_dict)
    ssnd.plot(mod_type="SVM", spl_type="NO_SH", feat_type="OPTICS_AS_DE", column_name="OATest",
              name="OPTICS_AS_DE NO SHADOW",
              **VEG_nosh_dict)

    plt.legend(
        # bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0
    )
    plt.title("SVM")
    plt.show()


class ShadowMain:

    def __init__(self):
        self.init_dfn = DirFileName(r"F:\ProjectSet\Shadow")
        self.raster_dfn = None
        self.sample_dfn = None

        self.model_dfn = None
        self.raster_fn = None
        self.sample_fn = None
        self.sample_csv_fn = None
        self.sample_csv_spl_fn = None

        self.category_names = ["IS", "VEG", "SOIL", "WAT"]
        self.category_sh_names = ["IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]

        self.sct = None
        self.grca = None

        self.optics_as_de_feats = {
            "OPTICS": (
                "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
                # "OPT_dis", "OPT_hom", "OPT_mean", "OPT_var"
            ),
            "AS": (
                "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2",
                "AS_SPAN",
                "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
                "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
                "AS_VV_hom", "AS_VV_mean", "AS_VV_var"
            ),
            "DE": (
                "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2",
                "DE_SPAN",
                "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
                "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
                "DE_VV_hom", "DE_VV_mean", "DE_VV_var"
            )
        }

        self.feat_deal = None
        self.ssn_mod_dirname = None

    def sampleToCsv(self, spl_fn=None, to_csv_fn=None):
        if spl_fn is None:
            spl_fn = self.sample_fn
        if to_csv_fn is None:
            to_csv_fn = self.sample_csv_fn
        df_train = pd.read_excel(spl_fn, sheet_name="Train")
        df_train["TEST"] = np.ones(len(df_train), dtype=int) * 1
        df_test = pd.read_excel(spl_fn, sheet_name="Test")
        df_test["TEST"] = np.ones(len(df_test), dtype=int) * 0
        df_shadow_test = pd.read_excel(spl_fn, sheet_name="ShadowTest")
        df_shadow_test["TEST"] = np.ones(len(df_shadow_test), dtype=int) * -1
        df = pd.concat([df_train, df_test, df_shadow_test])
        df.to_csv(self.sample_csv_fn, index=False)

    def sampling(self, csv_fn=None, gr=None, to_csv_fn=None):
        if csv_fn is None:
            csv_fn = self.sample_csv_fn
        if gr is None:
            gr = GDALRaster(self.raster_fn)
        if to_csv_fn is None:
            to_csv_fn = self.sample_csv_spl_fn
        samplingToCSV(csv_fn, gr, to_csv_fn)

    def sctCategoryColor(self):
        self.sct.sicAddCategory("IS", (255, 0, 0))
        self.sct.sicAddCategory("VEG", (0, 255, 0))
        self.sct.sicAddCategory("SOIL", (255, 255, 0))
        self.sct.sicAddCategory("WAT", (0, 0, 255))

    def testAcc(self, df, mod_dirname, to_csv_fn, to_txt_fn):
        self.grca = GDALRasterClassificationAccuracy()
        self.grca.addCategoryCode(IS=1, VEG=2, SOIL=3, WAT=4)
        self.grca.addDataFrame(df, c_column_name="CNAME")
        self.grca.openSaveCSVFileName(to_csv_fn)
        self.grca.openSaveCMFileName(to_txt_fn)
        self.grca.fitModelDirectory(mod_dirname)
        self.grca.closeSaveCSVFileName()
        self.grca.closeSaveCMFileName()

    def sctAddSVMRF(self):
        self.sct.addModelType("RF", trainRF)
        self.sct.addModelType("SVM", trainSvm)

    def sctAddSampleTypes(self):
        self.sct.addSampleType("SPL_NOSH", "IS", "VEG", "SOIL", "WAT")
        self.sct.addSampleType("SPL_SH", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH")

    def sctAddFeatureType(self, feat_type):
        if feat_type == "optics_as_de":
            self.sct.addFeatureType("OPTICS", *tuple(self.optics_as_de_feats["OPTICS"]))
            self.sct.addFeatureType("AS", *tuple(self.optics_as_de_feats["AS"]))
            self.sct.addFeatureType("DE", *tuple(self.optics_as_de_feats["DE"]))

    def featureDeal(self, obj_feat, feat_deal=None):
        if feat_deal is None:
            feat_deal = self.feat_deal
        feat_deal(obj_feat)

    def trainSampleNumber(self):
        spl_fn = self.sample_csv_spl_fn
        mod_dir = self.ssn_mod_dirname
        # spl_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_train_test_spl_s1.csv"

        csv_spl = CSVSamples(spl_fn)
        csv_spl.fieldNameCategory("CNAME")
        csv_spl.fieldNameTag("TAG")
        csv_spl.addCategoryNames(["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
        csv_spl.readData()
        csv_spl.is_trans = True
        csv_spl.is_scale_01 = True
        self.featureDeal(csv_spl)

        ssn = ShadowSampleNumber(mod_dir, n_category=4, category_names=["IS", "VEG", "SOIL", "WAT"])
        ssn.setSample(csv_spl)
        # ssn.addModelType("RF", trainRF)
        ssn.addModelType("SVM", trainSvm)
        ssn.sampleNumbers(1200, 200, 3)

        optics_feats = self.optics_as_de_feats["OPTICS"]
        as_feats = self.optics_as_de_feats["AS"]
        de_feats = self.optics_as_de_feats["DE"]
        ssn.addFeatureType("OPTICS", *tuple(optics_feats))
        ssn.addFeatureType("OPTICS_AS", *tuple(optics_feats + as_feats))
        ssn.addFeatureType("OPTICS_DE", *tuple(optics_feats + de_feats))
        ssn.addFeatureType("OPTICS_AS_DE", *tuple(optics_feats + as_feats + de_feats))

        ssn.sampleRandom()
        ssn.print()

        ssn.train()


if __name__ == "__main__":
    showSampleNumber()
