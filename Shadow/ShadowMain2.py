# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowMain.py
@Time    : 2023/7/8 10:31
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ShadowMain
-----------------------------------------------------------------------------"""

from matplotlib import pyplot as plt

from SRTCodes.SRTSample import CSVSamples
from SRTCodes.Utils import DirFileName
from Shadow.ShadowDraw import cal_10log10, SampleDrawBox, SampleDrawScatter, ShadowSampleNumberDraw
from Shadow.ShadowImdC import ShadowImageClassification
from Shadow.ShadowSample import ShadowSampleNumber
from Shadow.ShadowTraining import ShadowCategoryTraining, trainRF, trainSvm, ShadowCategoryTrainImdcOne, trainRF_nocv, \
    trainSvm_nocv

SAMPLE_FILENAME = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_train_test_spl_s1.csv"
FIG_DIR = r"F:\ProjectSet\Shadow\QingDao\mktu"


class ShadowQingDaoMain:

    def drawShadowBox(self):
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

    def drawShadowScatter(self):
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

    def shadowTraining(self):
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

    def shadowImdC(self):
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

    def trainSampleNumber(self):
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

    def showSampleNumber(self):
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
        ssnd.plot(mod_type="SVM", spl_type="SH", feat_type="OPTICS_AS_DE", column_name="OATest",
                  name="OPTICS_AS_DE SHADOW",
                  **VEG_sh_dict)
        ssnd.plot(mod_type="SVM", spl_type="NO_SH", feat_type="OPTICS_AS_DE", column_name="OATest",
                  name="OPTICS_AS_DE NO SHADOW",
                  **VEG_nosh_dict)

        plt.legend(
            # bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0
        )
        plt.title("SVM")
        plt.show()

    def main(self):
        return


class ShadowBeiJingMain:

    def __init__(self):
        self.init_dir = DirFileName(r"F:\ProjectSet\Shadow\BeiJing")
        self.mod_dir = self.init_dir.fn("Mods")

    def drawShadowBox(self):
        csv_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\6\sh_bj_spl6_4_2_spl.csv"
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
        csv_spl.featureCallBack("AS_SPAN", cal_10log10)
        csv_spl.featureCallBack("AS_Epsilon", cal_10log10)
        csv_spl.featureCallBack("DE_VV", cal_10log10)
        csv_spl.featureCallBack("DE_VH", cal_10log10)
        csv_spl.featureCallBack("DE_C11", cal_10log10)
        csv_spl.featureCallBack("DE_C22", cal_10log10)
        csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
        csv_spl.featureCallBack("DE_Lambda2", cal_10log10)
        csv_spl.featureCallBack("DE_SPAN", cal_10log10)
        csv_spl.featureCallBack("DE_Epsilon", cal_10log10)
        #
        # csv_spl.featureScaleMinMax("Blue", 299.76996, 2397.184)
        # csv_spl.featureScaleMinMax("Green", 345.83414, 2395.735)
        # csv_spl.featureScaleMinMax("Red", 177.79654, 2726.7026)
        # csv_spl.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        # csv_spl.featureScaleMinMax("NDVI", -0.6, 0.9)
        # csv_spl.featureScaleMinMax("NDWI", -0.7, 0.8)
        #
        # csv_spl.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        # csv_spl.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        # csv_spl.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        # csv_spl.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        # csv_spl.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        # csv_spl.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        # csv_spl.featureScaleMinMax("AS_SPAN", -25.869734, 10.284683)
        # csv_spl.featureScaleMinMax("AS_Epsilon", -10.0, 26.0)
        # # csv_spl.featureScaleMinMax("AS_Mu", 0.0, 1.0)
        # csv_spl.featureScaleMinMax("AS_RVI", 0, 2.76234)
        # # csv_spl.featureScaleMinMax("AS_m", 0.0, 1.0)
        # # csv_spl.featureScaleMinMax("AS_Beta", 0.0, 1.0)
        #
        # csv_spl.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        # csv_spl.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        # csv_spl.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        # csv_spl.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        # csv_spl.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        # csv_spl.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        # csv_spl.featureScaleMinMax("DE_SPAN", -27.869734, 13.284683)
        # csv_spl.featureScaleMinMax("DE_Epsilon", -6.0, 20.0)
        # # csv_spl.featureScaleMinMax("DE_Mu", 0.0, 1.0)
        # csv_spl.featureScaleMinMax("DE_RVI", 0, 2.76234)
        # # csv_spl.featureScaleMinMax("DE_m", 0.0, 1.0)
        # # csv_spl.featureScaleMinMax("DE_Beta", 0.0, 1.0)

        spl_draw_box = SampleDrawBox()
        spl_draw_box.setSample(csv_spl)

        spl_draw_box.colors("red", "darkred", "lightgreen", "darkgreen", "yellow", "y", "lightblue", "darkblue")

        # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
        spl_draw_box.addCategorys("IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "WAT")

        # "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        # "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        #     "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        # "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        #     "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
        spl_draw_box.addFeatures("NDVI", "NDWI")

        spl_draw_box.addTags("SELECT")

        spl_draw_box.print()

        spl_draw_box.fit()

        # plt.ylim([-1, 1])
        # plt.savefig()
        plt.show()

    def drawShadowScatter(self):
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

    def shadowTraining(self):
        mod_dir = r"F:\ProjectSet\Shadow\BeiJing\Mods"
        spl_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_train_test_spl_s1.csv"
        spl_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_train_test_spl_s2.csv"
        spl_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\7\sh_bj_spl7_1_rand_spl.csv"

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
        csv_spl.featureCallBack("AS_SPAN", cal_10log10)
        csv_spl.featureCallBack("AS_Epsilon", cal_10log10)
        csv_spl.featureCallBack("DE_VV", cal_10log10)
        csv_spl.featureCallBack("DE_VH", cal_10log10)
        csv_spl.featureCallBack("DE_C11", cal_10log10)
        csv_spl.featureCallBack("DE_C22", cal_10log10)
        csv_spl.featureCallBack("DE_Lambda1", cal_10log10)
        csv_spl.featureCallBack("DE_Lambda2", cal_10log10)
        csv_spl.featureCallBack("DE_SPAN", cal_10log10)
        csv_spl.featureCallBack("DE_Epsilon", cal_10log10)

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
        csv_spl.featureScaleMinMax("AS_SPAN", -25.869734, 10.284683)
        csv_spl.featureScaleMinMax("AS_Epsilon", -10.0, 26.0)
        # csv_spl.featureScaleMinMax("AS_Mu", 0.0, 1.0)
        csv_spl.featureScaleMinMax("AS_RVI", 0, 2.76234)
        # csv_spl.featureScaleMinMax("AS_m", 0.0, 1.0)
        # csv_spl.featureScaleMinMax("AS_Beta", 0.0, 1.0)

        csv_spl.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        csv_spl.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        csv_spl.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        csv_spl.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        csv_spl.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        csv_spl.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        csv_spl.featureScaleMinMax("DE_SPAN", -27.869734, 13.284683)
        csv_spl.featureScaleMinMax("DE_Epsilon", -6.0, 20.0)
        # csv_spl.featureScaleMinMax("DE_Mu", 0.0, 1.0)
        csv_spl.featureScaleMinMax("DE_RVI", 0, 2.76234)
        # csv_spl.featureScaleMinMax("DE_m", 0.0, 1.0)
        # csv_spl.featureScaleMinMax("DE_Beta", 0.0, 1.0)

        sct = ShadowCategoryTraining(mod_dir, n_category=4, category_names=["IS", "VEG", "SOIL", "WAT"])
        sct.setSample(csv_spl)
        # sct.addFrontFeatType("OPTICS", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI")
        sct.addModelType("RF", trainRF)
        sct.addModelType("SVM", trainSvm)

        # "NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH",
        sct.addSampleType("SPL_NOSH", "IS", "VEG", "SOIL", "WAT")
        sct.addSampleType("SPL_SH", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH")

        # "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        # "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        #   "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        # "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        #   "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
        # sct.addFeatureType("AS_SIGMA", "AS_VV", "AS_VH")
        # sct.addFeatureType("AS_C2", "AS_C11", "AS_C22", "AS_Epsilon")
        # sct.addFeatureType("AS_LAMD", "AS_Lambda1", "AS_Lambda2", "AS_SPAN")
        # sct.addFeatureType("DE_SIGMA", "DE_VV", "DE_VH")
        # sct.addFeatureType("DE_C2", "DE_C11", "DE_C22")
        # sct.addFeatureType("DE_LAMD", "DE_Lambda1", "DE_Lambda2")

        # sct.addFeatureType("OPTICS", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI")
        # sct.addFeatureType("AS", "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        #                    "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta")
        # sct.addFeatureType("DE", "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        #                    "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta")

        sct.addFeatureType("OPTICS", "Blue", "Green", "Red", "NIR", "NDVI", "NDWI")
        sct.addFeatureType("AS_SIGMA", "AS_VV", "AS_VH")
        sct.addFeatureType("AS_C2", "AS_C11", "AS_C22", "AS_Epsilon")
        sct.addFeatureType("AS_LAMD", "AS_Lambda1", "AS_Lambda2", "AS_SPAN")
        sct.addFeatureType("DE_SIGMA", "DE_VV", "DE_VH")
        sct.addFeatureType("DE_C2", "DE_C11", "DE_C22", "DE_Epsilon")
        sct.addFeatureType("DE_LAMD", "DE_Lambda1", "DE_Lambda2", "DE_SPAN")

        # "Select600ToTest", "Select1200ToTest", "NDWIWAT",  "Select",
        sct.addTagType("TAG", "Select600ToTest", "Select1200ToTest", "NDWIWAT", "Select")
        sct.print()

        sct.train()

        pass

    def shadowImdC(self):
        im_fn = r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_envi.dat"
        mod_dir = r"F:\ProjectSet\Shadow\BeiJing\Mods\20231017H215846"

        sic = ShadowImageClassification(im_fn, mod_dir)

        sic.featureCallBack("AS_VV", cal_10log10)
        sic.featureCallBack("AS_VH", cal_10log10)
        sic.featureCallBack("AS_C11", cal_10log10)
        sic.featureCallBack("AS_C22", cal_10log10)
        sic.featureCallBack("AS_Lambda1", cal_10log10)
        sic.featureCallBack("AS_Lambda2", cal_10log10)
        sic.featureCallBack("AS_SPAN", cal_10log10)
        sic.featureCallBack("AS_Epsilon", cal_10log10)
        sic.featureCallBack("DE_VV", cal_10log10)
        sic.featureCallBack("DE_VH", cal_10log10)
        sic.featureCallBack("DE_C11", cal_10log10)
        sic.featureCallBack("DE_C22", cal_10log10)
        sic.featureCallBack("DE_Lambda1", cal_10log10)
        sic.featureCallBack("DE_Lambda2", cal_10log10)
        sic.featureCallBack("DE_SPAN", cal_10log10)
        sic.featureCallBack("DE_Epsilon", cal_10log10)

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
        sic.featureScaleMinMax("AS_SPAN", -25.869734, 10.284683)
        sic.featureScaleMinMax("AS_Epsilon", -10.0, 26.0)
        # sic.featureScaleMinMax("AS_Mu", 0.0, 1.0)
        sic.featureScaleMinMax("AS_RVI", 0, 2.76234)
        # sic.featureScaleMinMax("AS_m", 0.0, 1.0)
        # sic.featureScaleMinMax("AS_Beta", 0.0, 1.0)

        sic.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        sic.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        sic.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        sic.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        sic.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        sic.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        sic.featureScaleMinMax("DE_SPAN", -27.869734, 13.284683)
        sic.featureScaleMinMax("DE_Epsilon", -6.0, 20.0)
        # csv_spl.featureScaleMinMax("DE_Mu", 0.0, 1.0)
        sic.featureScaleMinMax("DE_RVI", 0, 2.76234)
        # csv_spl.featureScaleMinMax("DE_m", 0.0, 1.0)
        # csv_spl.featureScaleMinMax("DE_Beta", 0.0, 1.0)

        sic.addCategory("IS", (255, 0, 0))
        sic.addCategory("VEG", (0, 255, 0))
        sic.addCategory("SOIL", (255, 255, 0))
        sic.addCategory("WAT", (0, 0, 255))

        sic.run()
        sic.print()
        pass

    def trainSampleNumber(self):
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

    def showSampleNumber(self):
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
        ssnd.plot(mod_type="SVM", spl_type="SH", feat_type="OPTICS_AS_DE", column_name="OATest",
                  name="OPTICS_AS_DE SHADOW",
                  **VEG_sh_dict)
        ssnd.plot(mod_type="SVM", spl_type="NO_SH", feat_type="OPTICS_AS_DE", column_name="OATest",
                  name="OPTICS_AS_DE NO SHADOW",
                  **VEG_nosh_dict)

        plt.legend(
            # bbox_to_anchor=(1.01, 0), loc=3, borderaxespad=0
        )
        plt.title("SVM")
        plt.show()

    def trainImdcOne(self):
        # spl_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\1\sh_bj_spl_summary2_600_train.csv"
        # spl_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\3\sh_bj_3_shnosh2.csv"
        spl_fn = r"F:\ProjectSet\Shadow\BeiJing\Samples\5\sh_bj_5_imdc1800_2.csv"
        raster_fn = r"F:\ProjectSet\Shadow\BeiJing\Image\3\BJ_SH3_envi.dat"

        scti = ShadowCategoryTrainImdcOne(self.mod_dir)
        scti.addGDALRaster(raster_fn)
        scti.initSIC(raster_fn)

        scti.addCSVFile(spl_fn, is_spl=False)
        scti.csv_spl.fieldNameCategory("CNAME")  # CNAME
        scti.csv_spl.fieldNameTag("TAG")
        scti.csv_spl.addCategoryNames(["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
        # scti.csv_spl.addCategoryNames(["SH", "NO_SH"])
        scti.csv_spl.readData()

        scti.featureCallBack("AS_VV", cal_10log10)
        scti.featureCallBack("AS_VH", cal_10log10)
        scti.featureCallBack("AS_C11", cal_10log10)
        scti.featureCallBack("AS_C22", cal_10log10)
        scti.featureCallBack("AS_Lambda1", cal_10log10)
        scti.featureCallBack("AS_Lambda2", cal_10log10)
        scti.featureCallBack("AS_SPAN", cal_10log10)
        scti.featureCallBack("AS_Epsilon", cal_10log10)
        scti.featureCallBack("DE_VV", cal_10log10)
        scti.featureCallBack("DE_VH", cal_10log10)
        scti.featureCallBack("DE_C11", cal_10log10)
        scti.featureCallBack("DE_C22", cal_10log10)
        scti.featureCallBack("DE_Lambda1", cal_10log10)
        scti.featureCallBack("DE_Lambda2", cal_10log10)
        scti.featureCallBack("DE_SPAN", cal_10log10)
        scti.featureCallBack("DE_Epsilon", cal_10log10)

        scti.featureScaleMinMax("Blue", 299.76996, 2397.184)
        scti.featureScaleMinMax("Green", 345.83414, 2395.735)
        scti.featureScaleMinMax("Red", 177.79654, 2726.7026)
        scti.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        scti.featureScaleMinMax("NDVI", -0.6, 0.9)
        scti.featureScaleMinMax("NDWI", -0.7, 0.8)

        scti.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        scti.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        scti.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        scti.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        scti.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        scti.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        scti.featureScaleMinMax("AS_SPAN", -25.869734, 10.284683)
        scti.featureScaleMinMax("AS_Epsilon", -10.0, 26.0)
        # scti.featureScaleMinMax("AS_Mu", 0.0, 1.0)
        scti.featureScaleMinMax("AS_RVI", 0, 2.76234)
        # scti.featureScaleMinMax("AS_m", 0.0, 1.0)
        # scti.featureScaleMinMax("AS_Beta", 0.0, 1.0)

        scti.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        scti.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        scti.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        scti.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        scti.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        scti.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        scti.featureScaleMinMax("DE_SPAN", -27.869734, 13.284683)
        scti.featureScaleMinMax("DE_Epsilon", -6.0, 20.0)
        # scti.featureScaleMinMax("DE_Mu", 0.0, 1.0)
        scti.featureScaleMinMax("DE_RVI", 0, 2.76234)
        # scti.featureScaleMinMax("DE_m", 0.0, 1.0)
        # scti.featureScaleMinMax("DE_Beta", 0.0, 1.0)

        scti.sicAddCategory("IS", (255, 0, 0))
        scti.sicAddCategory("VEG", (0, 255, 0))
        scti.sicAddCategory("SOIL", (255, 255, 0))
        scti.sicAddCategory("WAT", (0, 0, 255))

        # scti.sicAddCategory("SH", (255, 255,255))
        # scti.sicAddCategory("NO_SH", (0, 0, 0))

        scti.setSample()

        scti.fitFeatureNames(
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
            "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
            "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
            "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
            "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta"
        )
        # "NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
        scti.fitCategoryNames("IS", "VEG", "SOIL", "WAT")
        scti.fitCMNames("IS", "VEG", "SOIL", "WAT")

        # scti.fitFeatureNames(
        #     "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        #     "AS_VV", "AS_VH", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        #     "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        #     "DE_VV", "DE_VH", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        #     "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta"
        # )
        # scti.fitCMNames("SH", "NO_SH")

        scti.trainFunc(trainRF_nocv)
        # scti.trainFunc(trainSvm_nocv)

        scti.fit()

    def patchImage(self):
        # 使用一个点取出这个点附近范围内的影像
        # 116.448888, 39.826280
        im_fn = r""

    def main(self):
        self.shadowImdC()
        return


class ShadowMain:

    def __init__(self):
        self.release_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release")
        self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\Models")

    def shadowTraining(self):
        spl_fn = r"F:\ProjectSet\Shadow\QingDao\Sample\2\qd_spl2_3.csv"
        raster_fn = r"F:\ProjectSet\Shadow\Release\QingDaoImages\QD_SH.dat"
        model_dir = self.model_dfn.fn()
        model_name = "QingDao"

        sct = ShadowCategoryTraining(model_dir, model_name, n_category=4, category_names=["IS", "VEG", "SOIL", "WAT"])
        sct.initCSVSample(spl_fn, ["IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
        sct.initSIC(raster_fn)

        sct.featureScaleMinMax("Blue", 99.76996, 2397.184)
        sct.featureScaleMinMax("Green", 45.83414, 2395.735)
        sct.featureScaleMinMax("Red", 77.79654, 2726.7026)
        sct.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        sct.featureScaleMinMax("NDVI", -0.5007727, 0.7354284)
        sct.featureScaleMinMax("NDWI", -0.6572631, 0.7623875)
        sct.featureScaleMinMax("OPT_asm", 0.02124183, 0.998366)
        sct.featureScaleMinMax("OPT_con", 0.0, 169.74791)
        sct.featureScaleMinMax("OPT_cor", -0.036879253, 0.99688625)
        sct.featureScaleMinMax("OPT_dis", 0.0, 9.799746)
        sct.featureScaleMinMax("OPT_ent", 0.0, 3.8249474)
        sct.featureScaleMinMax("OPT_hom", 0.12091503, 0.998366)
        sct.featureScaleMinMax("OPT_mean", 4.941177, 53.7353)
        sct.featureScaleMinMax("OPT_var", 0.0, 236.09961)

        sct.featureCallBack("AS_VV", cal_10log10)
        sct.featureCallBack("AS_VH", cal_10log10)
        sct.featureCallBack("AS_C11", cal_10log10)
        sct.featureCallBack("AS_C22", cal_10log10)
        sct.featureCallBack("AS_Lambda1", cal_10log10)
        sct.featureCallBack("AS_Lambda2", cal_10log10)
        sct.featureCallBack("AS_SPAN", cal_10log10)
        sct.featureCallBack("AS_Epsilon", cal_10log10)
        sct.featureCallBack("DE_VV", cal_10log10)
        sct.featureCallBack("DE_VH", cal_10log10)
        sct.featureCallBack("DE_C11", cal_10log10)
        sct.featureCallBack("DE_C22", cal_10log10)
        sct.featureCallBack("DE_Lambda1", cal_10log10)
        sct.featureCallBack("DE_Lambda2", cal_10log10)
        sct.featureCallBack("DE_SPAN", cal_10log10)
        sct.featureCallBack("DE_Epsilon", cal_10log10)

        sct.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        sct.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        sct.featureScaleMinMax("AS_VHDVV", 0.0, 0.95164585)
        sct.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        # sct.featureScaleMinMax("AS_C12_imag", -0.3733027, -0.3733027)
        # sct.featureScaleMinMax("AS_C12_real", -0.31787363, -0.31787363)
        sct.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        sct.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        sct.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        # sct.featureScaleMinMax("AS_SPAN", 0.0, 0.0)
        sct.featureScaleMinMax("AS_Epsilon", 0.0, 35.12922)
        sct.featureScaleMinMax("AS_Mu", -0.7263123, 0.7037629)
        sct.featureScaleMinMax("AS_RVI", 0.07459847, 2.076324)
        sct.featureScaleMinMax("AS_m", 0.26469338, 0.97544414)
        sct.featureScaleMinMax("AS_Beta", 0.632338, 0.9869048)

        sct.featureScaleMinMax("AS_VH_asm", 0.02124183, 0.050653595)
        sct.featureScaleMinMax("AS_VH_con", 6.572378, 59.151405)
        sct.featureScaleMinMax("AS_VH_cor", 0.006340516, 0.86876196)
        sct.featureScaleMinMax("AS_VH_dis", 1.9767247, 5.8193297)
        sct.featureScaleMinMax("AS_VH_ent", 3.0939856, 3.8060431)
        sct.featureScaleMinMax("AS_VH_hom", 0.16666667, 0.40849674)
        sct.featureScaleMinMax("AS_VH_mean", 7.514706, 54.04412)
        sct.featureScaleMinMax("AS_VH_var", 5.9986033, 108.64137)
        sct.featureScaleMinMax("AS_VV_asm", 0.022875817, 0.050653595)
        sct.featureScaleMinMax("AS_VV_con", 4.5305123, 48.325462)
        sct.featureScaleMinMax("AS_VV_cor", 0.21234758, 0.88228023)
        sct.featureScaleMinMax("AS_VV_dis", 1.5990733, 5.22229)
        sct.featureScaleMinMax("AS_VV_ent", 3.1254923, 3.7871387)
        sct.featureScaleMinMax("AS_VV_hom", 0.18464053, 0.45261437)
        sct.featureScaleMinMax("AS_VV_mean", 8.544118, 51.573532)
        sct.featureScaleMinMax("AS_VV_var", 3.8744159, 96.8604)

        sct.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        sct.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        sct.featureScaleMinMax("DE_VHDVV", 0.0, 1.0289364)
        sct.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        # sct.featureScaleMinMax("DE_C12_imag", -0.80538744, -0.80538744)
        # sct.featureScaleMinMax("DE_C12_real", -0.4797325, 0.32061768)
        sct.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        sct.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        sct.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        # sct.featureScaleMinMax("DE_SPAN", 0.0, 0.0)
        sct.featureScaleMinMax("DE_Epsilon", 0.0, 21.882689)
        sct.featureScaleMinMax("DE_Mu", -0.6823329, 0.7723537)
        sct.featureScaleMinMax("DE_RVI", 0.0940072, 2.1935015)
        sct.featureScaleMinMax("DE_m", 0.24836189, 0.9705721)
        sct.featureScaleMinMax("DE_Beta", 0.6241778, 0.9852859)

        sct.featureScaleMinMax("DE_VH_asm", 0.022875817, 0.05392157)
        sct.featureScaleMinMax("DE_VH_con", 5.6798058, 51.11825)
        sct.featureScaleMinMax("DE_VH_cor", 0.12444292, 0.87177193)
        sct.featureScaleMinMax("DE_VH_dis", 1.8186697, 5.456009)
        sct.featureScaleMinMax("DE_VH_ent", 2.9679575, 3.7997417)
        sct.featureScaleMinMax("DE_VH_hom", 0.1748366, 0.42810458)
        sct.featureScaleMinMax("DE_VH_mean", 7.6176476, 55.176476)
        sct.featureScaleMinMax("DE_VH_var", 5.513511, 95.38374)
        sct.featureScaleMinMax("DE_VV_asm", 0.02124183, 0.057189543)
        sct.featureScaleMinMax("DE_VV_con", 5.0987973, 57.54357)
        sct.featureScaleMinMax("DE_VV_cor", 0.19514601, 0.88254523)
        sct.featureScaleMinMax("DE_VV_dis", 1.7117102, 5.6928787)
        sct.featureScaleMinMax("DE_VV_ent", 2.993163, 3.7997417)
        sct.featureScaleMinMax("DE_VV_hom", 0.17320262, 0.44444445)
        sct.featureScaleMinMax("DE_VV_mean", 6.4852943, 54.04412)
        sct.featureScaleMinMax("DE_VV_var", 4.44714, 111.17851)

        sct.sicAddCategory("IS", (255, 0, 0))
        sct.sicAddCategory("VEG", (0, 255, 0))
        sct.sicAddCategory("SOIL", (255, 255, 0))
        sct.sicAddCategory("WAT", (0, 0, 255))

        sct.setSample()

        sct.addModelType("RF", trainRF_nocv)
        sct.addModelType("SVM", trainSvm_nocv)

        #  "NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
        sct.addSampleType("SPL_NOSH", "IS", "VEG", "SOIL", "WAT")
        sct.addSampleType("SPL_SH", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH")

        # "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
        # "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
        # "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C12_imag", "AS_C12_real", "AS_C22", "AS_Lambda1", "AS_Lambda2",
        # "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
        # "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
        # "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        # "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C12_imag", "DE_C12_real", "DE_C22", "DE_Lambda1", "DE_Lambda2",
        # "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
        # "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
        # "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
        # "mean", "variance", "homogeneity", "contrast", "dissimilarity", "entropy", "second moment", "correlation"

        sct.addFeatureType(
            "OPTICS",
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
            "OPT_dis", "OPT_hom", "OPT_mean", "OPT_var"
        )
        sct.addFeatureType(
            "AS",
            "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2",
            "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
            "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
            "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        )
        sct.addFeatureType(
            "DE",
            "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2",
            "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
            "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
            "DE_VV_hom", "DE_VV_mean", "DE_VV_var"
        )

        #   "SELECT", "STATIFY_SH", "GE_MAP_RAND", "SEA", "NIR_SH",   "STATIFY_NOSH",
        sct.addTagType("TAG", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "SEA", "NIR_SH", "STATIFY_NOSH", )
        sct.print()

        sct.train()

        ...

    def main(self):
        self.shadowTraining()


if __name__ == "__main__":
    # ShadowQingDaoMain().main()
    ShadowMain().main()
