# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowMainBeiJing.py
@Time    : 2023/11/13 19:13
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowMainBeiJing
-----------------------------------------------------------------------------"""

import pandas as pd

from SRTCodes.Utils import DirFileName
from Shadow.ShadowDraw import cal_10log10
from Shadow.ShadowMain import ShadowMain
from Shadow.ShadowTraining import ShadowCategoryTraining


def bjFeatureDeal(obj_feat):
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
    obj_feat.featureScaleMinMax("Blue", 0.775024, 3811.798096)
    obj_feat.featureScaleMinMax("Green", 0.347000, 4143.411621)
    obj_feat.featureScaleMinMax("Red", 0.777344, 4356.097168)
    obj_feat.featureScaleMinMax("NIR", 0.653412, 5342.544922)
    obj_feat.featureScaleMinMax("NDVI", -0.460130, 0.885358)
    obj_feat.featureScaleMinMax("NDWI", -0.809526, 0.659520)
    obj_feat.featureScaleMinMax("OPT_asm", 0.000000, 0.916378)
    obj_feat.featureScaleMinMax("OPT_con", 0.000000, 257.148010)
    obj_feat.featureScaleMinMax("OPT_cor", -0.028123, 0.968763)
    obj_feat.featureScaleMinMax("OPT_dis", 0.000000, 12.008677)
    obj_feat.featureScaleMinMax("OPT_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("OPT_hom", 0.000000, 0.980469)
    obj_feat.featureScaleMinMax("OPT_mean", 0.000000, 60.310276)
    obj_feat.featureScaleMinMax("OPT_var", 0.000000, 419.989929)
    obj_feat.featureScaleMinMax("AS_VV", -21.276859, 17.139217)
    obj_feat.featureScaleMinMax("AS_VH", -26.283096, 6.622349)
    obj_feat.featureScaleMinMax("AS_VHDVV", 0.000066, 1.477573)
    obj_feat.featureScaleMinMax("AS_C11", -20.958439, 17.844374)
    obj_feat.featureScaleMinMax("AS_C12_imag", -2.929338, 2.199140)
    obj_feat.featureScaleMinMax("AS_C12_real", -7.886733, 2.941247)
    obj_feat.featureScaleMinMax("AS_C22", -24.533142, 7.302015)
    obj_feat.featureScaleMinMax("AS_Lambda1", -20.018930, 18.139565)
    obj_feat.featureScaleMinMax("AS_Lambda2", -25.640583, 1.551409)
    obj_feat.featureScaleMinMax("AS_SPAN", -18.507326, 18.200640)
    obj_feat.featureScaleMinMax("AS_Epsilon", -2.950454, 24.951567)
    obj_feat.featureScaleMinMax("AS_Mu", -0.960001, 0.895902)
    obj_feat.featureScaleMinMax("AS_RVI", 0.012637, 2.637578)
    obj_feat.featureScaleMinMax("AS_m", 0.129281, 0.999975)
    obj_feat.featureScaleMinMax("AS_Beta", 0.564640, 0.999987)
    obj_feat.featureScaleMinMax("AS_VH_asm", 0.000000, 0.942230)
    obj_feat.featureScaleMinMax("AS_VH_con", 0.000000, 120.384552)
    obj_feat.featureScaleMinMax("AS_VH_cor", -0.002791, 0.973579)
    obj_feat.featureScaleMinMax("AS_VH_dis", 0.000000, 8.382019)
    obj_feat.featureScaleMinMax("AS_VH_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("AS_VH_hom", 0.000000, 0.980469)
    obj_feat.featureScaleMinMax("AS_VH_mean", 0.000000, 60.547718)
    obj_feat.featureScaleMinMax("AS_VH_var", 0.000000, 309.779175)
    obj_feat.featureScaleMinMax("AS_VV_asm", 0.000000, 0.428757)
    obj_feat.featureScaleMinMax("AS_VV_con", 0.000000, 121.031212)
    obj_feat.featureScaleMinMax("AS_VV_cor", -0.001175, 0.948614)
    obj_feat.featureScaleMinMax("AS_VV_dis", 0.000000, 8.393357)
    obj_feat.featureScaleMinMax("AS_VV_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("AS_VV_hom", 0.000000, 0.707641)
    obj_feat.featureScaleMinMax("AS_VV_mean", 0.000000, 60.785160)
    obj_feat.featureScaleMinMax("AS_VV_var", 0.000000, 298.160095)
    obj_feat.featureScaleMinMax("DE_VV", -22.244835, 17.016680)
    obj_feat.featureScaleMinMax("DE_VH", -27.110813, 8.422224)
    obj_feat.featureScaleMinMax("DE_VHDVV", 0.000085, 1.831078)
    obj_feat.featureScaleMinMax("DE_C11", -20.984550, 17.693087)
    obj_feat.featureScaleMinMax("DE_C12_imag", -2.427857, 3.678876)
    obj_feat.featureScaleMinMax("DE_C12_real", -2.410979, 10.175308)
    obj_feat.featureScaleMinMax("DE_C22", -23.846441, 8.965826)
    obj_feat.featureScaleMinMax("DE_SPAN", -18.524240, 18.097307)
    obj_feat.featureScaleMinMax("DE_Lambda1", -20.194424, 18.113535)
    obj_feat.featureScaleMinMax("DE_Lambda2", -24.806219, 2.814495)
    obj_feat.featureScaleMinMax("DE_Epsilon", -3.059023, 22.729027)
    obj_feat.featureScaleMinMax("DE_Mu", -0.846598, 0.952367)
    obj_feat.featureScaleMinMax("DE_RVI", 0.015496, 2.682549)
    obj_feat.featureScaleMinMax("DE_m", 0.131804, 0.996071)
    obj_feat.featureScaleMinMax("DE_Beta", 0.565900, 0.998036)
    obj_feat.featureScaleMinMax("DE_VH_asm", 0.000000, 1.000000)
    obj_feat.featureScaleMinMax("DE_VH_con", 0.000000, 143.315582)
    obj_feat.featureScaleMinMax("DE_VH_cor", -0.003380, 1.000000)
    obj_feat.featureScaleMinMax("DE_VH_dis", 0.000000, 9.058996)
    obj_feat.featureScaleMinMax("DE_VH_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("DE_VH_hom", 0.000000, 1.000000)
    obj_feat.featureScaleMinMax("DE_VH_mean", 0.000000, 61.277348)
    obj_feat.featureScaleMinMax("DE_VH_var", 0.000000, 341.433167)
    obj_feat.featureScaleMinMax("DE_VV_asm", 0.000000, 0.506621)
    obj_feat.featureScaleMinMax("DE_VV_con", 0.000000, 130.093811)
    obj_feat.featureScaleMinMax("DE_VV_cor", -0.003181, 0.952050)
    obj_feat.featureScaleMinMax("DE_VV_dis", 0.000000, 8.681627)
    obj_feat.featureScaleMinMax("DE_VV_ent", 0.000000, 3.891819)
    obj_feat.featureScaleMinMax("DE_VV_hom", 0.000000, 0.772312)
    obj_feat.featureScaleMinMax("DE_VV_mean", 0.000000, 60.310276)
    obj_feat.featureScaleMinMax("DE_VV_var", 0.000000, 316.407379)


class ShadowMainBJ(ShadowMain):

    def __init__(self):
        super().__init__()
        self.model_name = "BeiJing"

        self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingImages")
        self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingSamples")
        self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\BeiJing\Mods")

        self.raster_fn = self.raster_dfn.fn("SH_BJ_envi.dat")
        self.sample_fn = self.sample_dfn.fn("BeiJingSamples.xlsx")
        self.sample_csv_fn = self.sample_dfn.fn("sh_bj_sample.csv")
        self.sample_csv_spl_fn = self.sample_dfn.fn("sh_bj_sample_spl.csv")

        self.feat_deal = bjFeatureDeal
        self.ssn_mod_dirname=self.init_dfn.fn(r"BeiJing\SampleNumber\Mods")

    def shadowTraining(self):
        # spl_fn = self.sample_csv_spl_fn
        # spl_fn = r"F:\ProjectSet\Shadow\Analysis\5\sh_bj_sample_spl2.csv"
        spl_fn = r"F:\ProjectSet\Shadow\Analysis\8\test\bj_train_data_t2_spl.csv"
        raster_fn = self.raster_fn
        model_dir = self.model_dfn.fn()
        model_name = self.model_name

        self.sct = ShadowCategoryTraining(model_dir, model_name, n_category=4,
                                          category_names=["IS", "VEG", "SOIL", "WAT"])
        self.sct.initCSVSample(spl_fn, ["IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
        self.sct.initSIC(raster_fn)
        bjFeatureDeal(self.sct)

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

        #   "Select1200ToTest", "Select", "Select600ToTest", "NDWIRANDOM", "NDWIWAT",
        self.sct.addTagType("TAG", "Select1200ToTest", "Select", "Select600ToTest", "NDWIRANDOM", "NDWIWAT", )
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
    sm_bj = ShadowMainBJ()
    # sm_bj.sampleToCsv()
    # sm_bj.sampling()
    # sm_bj.shadowTraining()
    # sm_bj.testImdc()
    sm_bj.trainSampleNumber()
    pass


if __name__ == "__main__":
    # python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowMainBeiJing import main; main()"
    main()
