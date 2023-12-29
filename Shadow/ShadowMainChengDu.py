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


def cdFeatureDeal(obj_feat):
    obj_feat.featureCallBack("AS_VV", cal_10log10)
    obj_feat.featureCallBack("AS_VH", cal_10log10)
    obj_feat.featureCallBack("AS_C11", cal_10log10)
    obj_feat.featureCallBack("AS_C22", cal_10log10)
    obj_feat.featureCallBack("AS_Lambda1", cal_10log10)
    obj_feat.featureCallBack("AS_Lambda2", cal_10log10)
    obj_feat.featureCallBack("AS_SPAN", cal_10log10)
    obj_feat.featureCallBack("AS_Epsilon", cal_10log10)
    obj_feat.featureCallBack("DE_VV", cal_10log10)
    obj_feat.featureCallBack("DE_VH", cal_10log10)
    obj_feat.featureCallBack("DE_C11", cal_10log10)
    obj_feat.featureCallBack("DE_C22", cal_10log10)
    obj_feat.featureCallBack("DE_Lambda1", cal_10log10)
    obj_feat.featureCallBack("DE_Lambda2", cal_10log10)
    obj_feat.featureCallBack("DE_SPAN", cal_10log10)
    obj_feat.featureCallBack("DE_Epsilon", cal_10log10)
    obj_feat.featureScaleMinMax("Blue", 0.108612, 2833.325195)
    obj_feat.featureScaleMinMax("Green", 0.730469, 3284.211914)
    obj_feat.featureScaleMinMax("Red", 0.364441, 3528.857910)
    obj_feat.featureScaleMinMax("NIR", 0.480164, 4164.366211)
    obj_feat.featureScaleMinMax("NDVI", -0.281195, 0.775548)
    obj_feat.featureScaleMinMax("NDWI", -0.683179, 0.359006)
    obj_feat.featureScaleMinMax("OPT_asm", 0.000000, 0.682763)
    obj_feat.featureScaleMinMax("OPT_con", 0.000000, 200.413773)
    obj_feat.featureScaleMinMax("OPT_cor", -0.003125, 0.962777)
    obj_feat.featureScaleMinMax("OPT_dis", 0.000000, 10.269228)
    obj_feat.featureScaleMinMax("OPT_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("OPT_hom", 0.000000, 0.916378)
    obj_feat.featureScaleMinMax("OPT_mean", 0.000000, 59.255516)
    obj_feat.featureScaleMinMax("OPT_var", 0.000000, 337.672028)
    obj_feat.featureScaleMinMax("AS_VV", -20.900673, 16.643545)
    obj_feat.featureScaleMinMax("AS_VH", -26.645557, 7.060156)
    obj_feat.featureScaleMinMax("AS_VHDVV", 0.000149, 1.984372)
    obj_feat.featureScaleMinMax("AS_C11", -21.075790, 17.110327)
    obj_feat.featureScaleMinMax("AS_C12_imag", -2.651620, 2.423286)
    obj_feat.featureScaleMinMax("AS_C12_real", -5.776805, 3.410573)
    obj_feat.featureScaleMinMax("AS_C22", -25.058722, 7.454190)
    obj_feat.featureScaleMinMax("AS_Lambda1", -20.538422, 17.272667)
    obj_feat.featureScaleMinMax("AS_Lambda2", -25.871984, 2.598621)
    obj_feat.featureScaleMinMax("AS_SPAN", -19.152206, 17.335613)
    obj_feat.featureScaleMinMax("AS_Epsilon", -3.575741, 24.316652)
    obj_feat.featureScaleMinMax("AS_Mu", -0.936081, 0.906247)
    obj_feat.featureScaleMinMax("AS_RVI", 0.014749, 2.773232)
    obj_feat.featureScaleMinMax("AS_m", 0.114727, 0.996446)
    obj_feat.featureScaleMinMax("AS_Beta", 0.557366, 0.998223)
    obj_feat.featureScaleMinMax("AS_VV_asm", 0.000000, 0.633058)
    obj_feat.featureScaleMinMax("AS_VV_con", 0.000000, 146.806534)
    obj_feat.featureScaleMinMax("AS_VV_cor", -0.003414, 0.946148)
    obj_feat.featureScaleMinMax("AS_VV_dis", 0.000000, 9.265110)
    obj_feat.featureScaleMinMax("AS_VV_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("AS_VV_hom", 0.000000, 0.836151)
    obj_feat.featureScaleMinMax("AS_VV_mean", 0.000000, 61.523441)
    obj_feat.featureScaleMinMax("AS_VV_var", 0.000000, 337.570496)
    obj_feat.featureScaleMinMax("AS_VH_asm", 0.000000, 0.829619)
    obj_feat.featureScaleMinMax("AS_VH_con", 0.000000, 139.858276)
    obj_feat.featureScaleMinMax("AS_VH_cor", -0.002550, 0.957325)
    obj_feat.featureScaleMinMax("AS_VH_dis", 0.000000, 9.079297)
    obj_feat.featureScaleMinMax("AS_VH_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("AS_VH_hom", 0.000000, 0.930923)
    obj_feat.featureScaleMinMax("AS_VH_mean", 0.000000, 60.547718)
    obj_feat.featureScaleMinMax("AS_VH_var", 0.000000, 334.443359)
    obj_feat.featureScaleMinMax("DE_VV", -22.628164, 16.044880)
    obj_feat.featureScaleMinMax("DE_VH", -27.321905, 7.506963)
    obj_feat.featureScaleMinMax("DE_VHDVV", 0.000097, 2.286051)
    obj_feat.featureScaleMinMax("DE_C11", -21.412586, 16.697243)
    obj_feat.featureScaleMinMax("DE_C12_imag", -2.605879, 2.580937)
    obj_feat.featureScaleMinMax("DE_C12_real", -4.237278, 4.949156)
    obj_feat.featureScaleMinMax("DE_C22", -24.794399, 8.057488)
    obj_feat.featureScaleMinMax("DE_Lambda1", -20.405666, 17.049181)
    obj_feat.featureScaleMinMax("DE_Lambda2", -25.859194, 3.229606)
    obj_feat.featureScaleMinMax("DE_SPAN", -18.702457, 17.098742)
    obj_feat.featureScaleMinMax("DE_Epsilon", -4.977984, 24.332211)
    obj_feat.featureScaleMinMax("DE_Mu", -0.921154, 0.936814)
    obj_feat.featureScaleMinMax("DE_RVI", 0.014615, 3.027073)
    obj_feat.featureScaleMinMax("DE_Beta", 0.066218, 0.999969)
    obj_feat.featureScaleMinMax("DE_m", 0.533109, 0.999985)
    obj_feat.featureScaleMinMax("DE_VH_asm", 0.000000, 0.506621)
    obj_feat.featureScaleMinMax("DE_VH_con", 0.000000, 145.302383)
    obj_feat.featureScaleMinMax("DE_VH_cor", -0.001322, 0.949018)
    obj_feat.featureScaleMinMax("DE_VH_dis", 0.000000, 9.225253)
    obj_feat.featureScaleMinMax("DE_VH_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("DE_VH_hom", 0.000000, 0.760149)
    obj_feat.featureScaleMinMax("DE_VH_mean", 0.000000, 61.277348)
    obj_feat.featureScaleMinMax("DE_VH_var", 0.000000, 334.318817)
    obj_feat.featureScaleMinMax("DE_VV_asm", 0.000000, 0.953537)
    obj_feat.featureScaleMinMax("DE_VV_con", 0.000000, 182.110428)
    obj_feat.featureScaleMinMax("DE_VV_cor", -0.003191, 0.969524)
    obj_feat.featureScaleMinMax("DE_VV_dis", 0.000000, 10.180984)
    obj_feat.featureScaleMinMax("DE_VV_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("DE_VV_hom", 0.000000, 0.980469)
    obj_feat.featureScaleMinMax("DE_VV_mean", 0.000000, 62.015629)
    obj_feat.featureScaleMinMax("DE_VV_var", 0.000000, 399.645813)


class ShadowMainCD(ShadowMain):

    def __init__(self):
        super().__init__()
        self.model_name = "ChengDu"

        self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\ChengDuImages")
        self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\ChengDuSamples")
        self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\ChengDu\Mods")

        self.raster_fn = self.raster_dfn.fn("SH_CD_envi.dat")
        self.sample_fn = self.sample_dfn.fn("ChengDuSamples.xlsx")
        self.sample_csv_fn = self.sample_dfn.fn("sh_cd_sample.csv")
        self.sample_csv_spl_fn = self.sample_dfn.fn("sh_cd_sample_spl.csv")

        self.feat_deal = cdFeatureDeal
        self.ssn_mod_dirname=self.init_dfn.fn(r"ChengDu\SampleNumber\Mods")

    def shadowTraining(self):
        # spl_fn = self.sample_csv_spl_fn
        # spl_fn = r"F:\ProjectSet\Shadow\Analysis\5\sh_cd_sample_spl2.csv"
        spl_fn = r"F:\ProjectSet\Shadow\Analysis\8\chengdu\sh_cd_sample_spl_2.csv"
        raster_fn = self.raster_fn
        model_dir = self.model_dfn.fn()
        model_name = self.model_name

        self.sct = ShadowCategoryTraining(model_dir, model_name, n_category=4,
                                          category_names=["IS", "VEG", "SOIL", "WAT"])
        self.sct.initCSVSample(spl_fn, ["IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
        self.sct.initSIC(raster_fn)
        cdFeatureDeal(self.sct)

        self.sctCategoryColor()

        self.sct.setSample()

        self.sctAddSVMRF()

        #  "NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
        # self.sct.addSampleType("SPL_NOSH", "IS", "VEG", "SOIL", "WAT")
        # self.sct.addSampleType("SPL_SH", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH")
        self.sctAddSampleTypes()

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
        self.sctAddFeatureType("optics_as_de")

        # "Random3000", "NDWIRandom", "Select",
        self.sct.addTagType("TAG","Random3000", "NDWIRandom", "Select")
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
    sm_cd = ShadowMainCD()
    # sm_cd.sampleToCsv()
    # sm_cd.sampling()
    # sm_cd.shadowTraining()
    # sm_cd.testImdc()
    sm_cd.trainSampleNumber()
    pass


if __name__ == "__main__":
    # python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowMainChengDu import main; main()"
    main()
