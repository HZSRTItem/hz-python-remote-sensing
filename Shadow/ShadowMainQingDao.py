# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowMainQingDao.py
@Time    : 2023/11/13 10:20
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowMainQingDao
-----------------------------------------------------------------------------"""

import pandas as pd

from SRTCodes.Utils import DirFileName
from Shadow.ShadowDraw import cal_10log10
from Shadow.ShadowMain import ShadowMain
from Shadow.ShadowTraining import ShadowCategoryTraining, trainRF, trainSvm


class ShadowMainQD(ShadowMain):

    def __init__(self):
        super().__init__()
        self.model_name = "QingDao"

        self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\QingDaoImages")
        self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\QingDaoSamples")
        self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\QingDao\Mods")

        self.raster_fn = self.raster_dfn.fn("SH_QD_envi.dat")
        self.sample_fn = self.sample_dfn.fn("QingDaoSamples.xlsx")
        self.sample_csv_fn = self.sample_dfn.fn("sh_qd_sample.csv")
        self.sample_csv_spl_fn = self.sample_dfn.fn("sh_qd_sample_spl.csv")

    def shadowTraining(self):
        spl_fn = self.sample_csv_spl_fn
        raster_fn = self.raster_fn
        model_dir = self.model_dfn.fn()
        model_name = self.model_name

        self.sct = ShadowCategoryTraining(model_dir, model_name, n_category=4,
                                          category_names=["IS", "VEG", "SOIL", "WAT"])
        self.sct.initCSVSample(spl_fn, ["IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
        self.sct.initSIC(raster_fn)

        self.sct.featureScaleMinMax("Blue", 99.76996, 2397.184)
        self.sct.featureScaleMinMax("Green", 45.83414, 2395.735)
        self.sct.featureScaleMinMax("Red", 77.79654, 2726.7026)
        self.sct.featureScaleMinMax("NIR", 87.66086, 3498.4321)
        self.sct.featureScaleMinMax("NDVI", -0.5007727, 0.7354284)
        self.sct.featureScaleMinMax("NDWI", -0.6572631, 0.7623875)
        self.sct.featureScaleMinMax("OPT_asm", 0.02124183, 0.998366)
        self.sct.featureScaleMinMax("OPT_con", 0.0, 169.74791)
        self.sct.featureScaleMinMax("OPT_cor", -0.036879253, 0.99688625)
        self.sct.featureScaleMinMax("OPT_dis", 0.0, 9.799746)
        self.sct.featureScaleMinMax("OPT_ent", 0.0, 3.8249474)
        self.sct.featureScaleMinMax("OPT_hom", 0.12091503, 0.998366)
        self.sct.featureScaleMinMax("OPT_mean", 4.941177, 53.7353)
        self.sct.featureScaleMinMax("OPT_var", 0.0, 236.09961)

        self.sct.featureCallBack("AS_VV", cal_10log10)
        self.sct.featureCallBack("AS_VH", cal_10log10)
        self.sct.featureCallBack("AS_C11", cal_10log10)
        self.sct.featureCallBack("AS_C22", cal_10log10)
        self.sct.featureCallBack("AS_Lambda1", cal_10log10)
        self.sct.featureCallBack("AS_Lambda2", cal_10log10)
        self.sct.featureCallBack("AS_SPAN", cal_10log10)
        self.sct.featureCallBack("AS_Epsilon", cal_10log10)
        self.sct.featureCallBack("DE_VV", cal_10log10)
        self.sct.featureCallBack("DE_VH", cal_10log10)
        self.sct.featureCallBack("DE_C11", cal_10log10)
        self.sct.featureCallBack("DE_C22", cal_10log10)
        self.sct.featureCallBack("DE_Lambda1", cal_10log10)
        self.sct.featureCallBack("DE_Lambda2", cal_10log10)
        self.sct.featureCallBack("DE_SPAN", cal_10log10)
        self.sct.featureCallBack("DE_Epsilon", cal_10log10)

        self.sct.featureScaleMinMax("AS_VV", -24.609674, 5.9092603)
        self.sct.featureScaleMinMax("AS_VH", -31.865038, -5.2615275)
        self.sct.featureScaleMinMax("AS_VHDVV", 0.0, 0.95164585)
        self.sct.featureScaleMinMax("AS_C11", -22.61998, 5.8634768)
        self.sct.featureScaleMinMax("AS_C22", -28.579813, -5.2111626)
        self.sct.featureScaleMinMax("AS_Lambda1", -21.955856, 6.124724)
        self.sct.featureScaleMinMax("AS_Lambda2", -29.869734, -8.284683)
        self.sct.featureScaleMinMax("AS_SPAN", -22.58362, 6.97997)
        self.sct.featureScaleMinMax("AS_Epsilon", 0.0, 35.12922)
        self.sct.featureScaleMinMax("AS_Mu", -0.7263123, 0.7037629)
        self.sct.featureScaleMinMax("AS_RVI", 0.07459847, 2.076324)
        self.sct.featureScaleMinMax("AS_m", 0.26469338, 0.97544414)
        self.sct.featureScaleMinMax("AS_Beta", 0.632338, 0.9869048)

        self.sct.featureScaleMinMax("AS_VH_asm", 0.02124183, 0.050653595)
        self.sct.featureScaleMinMax("AS_VH_con", 6.572378, 59.151405)
        self.sct.featureScaleMinMax("AS_VH_cor", 0.006340516, 0.86876196)
        self.sct.featureScaleMinMax("AS_VH_dis", 1.9767247, 5.8193297)
        self.sct.featureScaleMinMax("AS_VH_ent", 3.0939856, 3.8060431)
        self.sct.featureScaleMinMax("AS_VH_hom", 0.16666667, 0.40849674)
        self.sct.featureScaleMinMax("AS_VH_mean", 7.514706, 54.04412)
        self.sct.featureScaleMinMax("AS_VH_var", 5.9986033, 108.64137)
        self.sct.featureScaleMinMax("AS_VV_asm", 0.022875817, 0.050653595)
        self.sct.featureScaleMinMax("AS_VV_con", 4.5305123, 48.325462)
        self.sct.featureScaleMinMax("AS_VV_cor", 0.21234758, 0.88228023)
        self.sct.featureScaleMinMax("AS_VV_dis", 1.5990733, 5.22229)
        self.sct.featureScaleMinMax("AS_VV_ent", 3.1254923, 3.7871387)
        self.sct.featureScaleMinMax("AS_VV_hom", 0.18464053, 0.45261437)
        self.sct.featureScaleMinMax("AS_VV_mean", 8.544118, 51.573532)
        self.sct.featureScaleMinMax("AS_VV_var", 3.8744159, 96.8604)

        self.sct.featureScaleMinMax("DE_VV", -27.851603, 5.094706)
        self.sct.featureScaleMinMax("DE_VH", -35.427082, -5.4092093)
        self.sct.featureScaleMinMax("DE_VHDVV", 0.0, 1.0289364)
        self.sct.featureScaleMinMax("DE_C11", -26.245598, 4.9907513)
        self.sct.featureScaleMinMax("DE_C22", -32.042320, -5.322515)
        self.sct.featureScaleMinMax("DE_Lambda1", -25.503738, 5.2980003)
        self.sct.featureScaleMinMax("DE_Lambda2", -33.442368, -8.68537)
        self.sct.featureScaleMinMax("DE_SPAN", -24.81076, 4.82663)
        self.sct.featureScaleMinMax("DE_Epsilon", 0.0, 21.882689)
        self.sct.featureScaleMinMax("DE_Mu", -0.6823329, 0.7723537)
        self.sct.featureScaleMinMax("DE_RVI", 0.0940072, 2.1935015)
        self.sct.featureScaleMinMax("DE_m", 0.24836189, 0.9705721)
        self.sct.featureScaleMinMax("DE_Beta", 0.6241778, 0.9852859)

        self.sct.featureScaleMinMax("DE_VH_asm", 0.022875817, 0.05392157)
        self.sct.featureScaleMinMax("DE_VH_con", 5.6798058, 51.11825)
        self.sct.featureScaleMinMax("DE_VH_cor", 0.12444292, 0.87177193)
        self.sct.featureScaleMinMax("DE_VH_dis", 1.8186697, 5.456009)
        self.sct.featureScaleMinMax("DE_VH_ent", 2.9679575, 3.7997417)
        self.sct.featureScaleMinMax("DE_VH_hom", 0.1748366, 0.42810458)
        self.sct.featureScaleMinMax("DE_VH_mean", 7.6176476, 55.176476)
        self.sct.featureScaleMinMax("DE_VH_var", 5.513511, 95.38374)
        self.sct.featureScaleMinMax("DE_VV_asm", 0.02124183, 0.057189543)
        self.sct.featureScaleMinMax("DE_VV_con", 5.0987973, 57.54357)
        self.sct.featureScaleMinMax("DE_VV_cor", 0.19514601, 0.88254523)
        self.sct.featureScaleMinMax("DE_VV_dis", 1.7117102, 5.6928787)
        self.sct.featureScaleMinMax("DE_VV_ent", 2.993163, 3.7997417)
        self.sct.featureScaleMinMax("DE_VV_hom", 0.17320262, 0.44444445)
        self.sct.featureScaleMinMax("DE_VV_mean", 6.4852943, 54.04412)
        self.sct.featureScaleMinMax("DE_VV_var", 4.44714, 111.17851)

        self.sctCategoryColor()

        self.sct.setSample()

        self.sct.addModelType("RF", trainRF)
        self.sct.addModelType("SVM", trainSvm)

        #  "NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
        self.sct.addSampleType("SPL_NOSH", "IS", "VEG", "SOIL", "WAT")
        self.sct.addSampleType("SPL_SH", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH")

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

        self.sct.addFeatureType(
            "OPTICS",
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
            "OPT_dis", "OPT_hom", "OPT_mean", "OPT_var"
        )
        self.sct.addFeatureType(
            "AS",
            "AS_VV", "AS_VH", "AS_VHDVV", "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2",
            "AS_SPAN", "AS_Epsilon", "AS_Mu", "AS_RVI", "AS_m", "AS_Beta",
            "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
            "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
        )
        self.sct.addFeatureType(
            "DE",
            "DE_VV", "DE_VH", "DE_VHDVV", "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2",
            "DE_SPAN", "DE_Epsilon", "DE_Mu", "DE_RVI", "DE_m", "DE_Beta",
            "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
            "DE_VV_hom", "DE_VV_mean", "DE_VV_var"
        )

        # "SELECT", "STATIFY_SH", "GE_MAP_RAND", "SEA", "NIR_SH",   "STATIFY_NOSH",
        self.sct.addTagType("TAG", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "SEA", "NIR_SH", "STATIFY_NOSH", )
        self.sct.print()

        self.sct.train()

    def testImdc(self):
        # 使用新的测试样本测试分类影像的精度
        model = "20230707H200910"
        mod_dirname = self.model_dfn.fn(model)

        sheet_name = "Test"
        to_csv_fn = self.model_dfn.fn(model, model + "_acc.csv")
        to_txt_fn = self.model_dfn.fn(model, model + "_cm.txt")

        df = pd.read_excel(self.sample_fn, sheet_name=sheet_name)
        for i in range(len(df)):
            cname = str(df["CNAME"][i])
            if cname == "IS_SH":
                cname = "IS"
            elif cname == "VEG_SH":
                cname = "VEG"
            elif cname == "SOIL_SH":
                cname = "SOIL"
            elif cname == "WAT_SH":
                cname = "WAT"
            df.loc[i, "CNAME"] = cname

        self.testAcc(df, mod_dirname, to_csv_fn, to_txt_fn)


def main():
    sm_qd = ShadowMainQD()
    sm_qd.sampleToCsv()
    sm_qd.sampling()
    # sm_qd.shadowTraining()
    # sm_qd.testImdc()
    pass


if __name__ == "__main__":
    main()
