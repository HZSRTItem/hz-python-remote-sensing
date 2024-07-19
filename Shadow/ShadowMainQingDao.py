# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowMainQingDao.py
@Time    : 2023/11/13 10:20
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowMainQingDao
-----------------------------------------------------------------------------"""

import pandas as pd

from SRTCodes.Utils import DirFileName
from Shadow.ShadowDraw import cal_10log10
from Shadow.ShadowMain import ShadowMain
from Shadow.ShadowTraining import ShadowCategoryTraining, trainRF, trainSvm


def qdFeatureDeal(obj_feat):
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

    obj_feat.featureScaleMinMax("Blue", 0.908081, 3433.637207)
    obj_feat.featureScaleMinMax("Green", 0.762482, 3698.388428)
    obj_feat.featureScaleMinMax("Red", 0.996323, 3894.453857)
    obj_feat.featureScaleMinMax("NIR", 0.464844, 4722.584961)
    obj_feat.featureScaleMinMax("NDVI", -0.560402, 0.810093)
    obj_feat.featureScaleMinMax("NDWI", -0.716581, 0.804559)
    obj_feat.featureScaleMinMax("OPT_asm", 0.000000, 1.000000)
    obj_feat.featureScaleMinMax("OPT_con", 0.000000, 272.015747)
    obj_feat.featureScaleMinMax("OPT_cor", -0.111892, 1.000000)
    obj_feat.featureScaleMinMax("OPT_dis", 0.000000, 12.370493)
    obj_feat.featureScaleMinMax("OPT_ent", 0.000000, 3.856454)
    obj_feat.featureScaleMinMax("OPT_hom", 0.000000, 1.000000)
    obj_feat.featureScaleMinMax("OPT_mean", 0.000000, 59.597950)
    obj_feat.featureScaleMinMax("OPT_var", 0.000000, 396.725067)
    obj_feat.featureScaleMinMax("AS_VV", -28.936602, 15.659369)
    obj_feat.featureScaleMinMax("AS_VH", -35.467525, 5.524069)
    obj_feat.featureScaleMinMax("AS_VHDVV", 0.000127, 2.479116)
    obj_feat.featureScaleMinMax("AS_C11", -25.438517, 15.586018)
    obj_feat.featureScaleMinMax("AS_C12_imag", -1.635573, 1.470299)
    obj_feat.featureScaleMinMax("AS_C12_real", -2.518003, 3.200973)
    obj_feat.featureScaleMinMax("AS_C22", -31.111473, 5.534301)
    obj_feat.featureScaleMinMax("AS_Lambda1", -24.523169, 15.957440)
    obj_feat.featureScaleMinMax("AS_Lambda2", -32.416134, 0.506478)
    obj_feat.featureScaleMinMax("AS_SPAN", -23.015398, 16.033403)
    obj_feat.featureScaleMinMax("AS_Epsilon", -3.910793, 24.427685)
    obj_feat.featureScaleMinMax("AS_Mu", -0.919362, 0.912376)
    obj_feat.featureScaleMinMax("AS_RVI", 0.014391, 2.848592)
    obj_feat.featureScaleMinMax("AS_m", 0.112938, 0.999953)
    obj_feat.featureScaleMinMax("AS_Beta", 0.556474, 0.999976)
    obj_feat.featureScaleMinMax("AS_VV_asm", 0.000000, 0.359939)
    obj_feat.featureScaleMinMax("AS_VV_con", 0.000000, 88.446480)
    obj_feat.featureScaleMinMax("AS_VV_cor", -0.000472, 0.935039)
    obj_feat.featureScaleMinMax("AS_VV_dis", 0.000000, 7.022758)
    obj_feat.featureScaleMinMax("AS_VV_ent", 0.000000, 3.841390)
    obj_feat.featureScaleMinMax("AS_VV_hom", 0.000000, 0.669300)
    obj_feat.featureScaleMinMax("AS_VV_mean", 0.000000, 61.031254)
    obj_feat.featureScaleMinMax("AS_VV_var", 0.000000, 195.577713)
    obj_feat.featureScaleMinMax("AS_VH_asm", 0.000000, 0.318144)
    obj_feat.featureScaleMinMax("AS_VH_con", 0.000000, 81.502403)
    obj_feat.featureScaleMinMax("AS_VH_cor", -0.092647, 0.927022)
    obj_feat.featureScaleMinMax("AS_VH_dis", 0.000000, 6.799606)
    obj_feat.featureScaleMinMax("AS_VH_ent", 0.000000, 3.841390)
    obj_feat.featureScaleMinMax("AS_VH_hom", 0.000000, 0.640685)
    obj_feat.featureScaleMinMax("AS_VH_mean", 0.000000, 60.785160)
    obj_feat.featureScaleMinMax("AS_VH_var", 0.000000, 173.091125)
    obj_feat.featureScaleMinMax("DE_VV", -30.057774, 14.817512)
    obj_feat.featureScaleMinMax("DE_VH", -37.066406, 6.256181)
    obj_feat.featureScaleMinMax("DE_VHDVV", 0.000113, 2.375435)
    obj_feat.featureScaleMinMax("DE_C11", -28.367586, 14.703403)
    obj_feat.featureScaleMinMax("DE_C12_imag", -1.406325, 2.041579)
    obj_feat.featureScaleMinMax("DE_C12_real", -1.868385, 3.706888)
    obj_feat.featureScaleMinMax("DE_C22", -33.604973, 6.280687)
    obj_feat.featureScaleMinMax("DE_Lambda1", -27.306553, 15.156054)
    obj_feat.featureScaleMinMax("DE_Lambda2", -34.749477, 1.366678)
    obj_feat.featureScaleMinMax("DE_SPAN", -25.774284, 15.190010)
    obj_feat.featureScaleMinMax("DE_Epsilon", -4.403823, 23.783085)
    obj_feat.featureScaleMinMax("DE_Mu", -0.899417, 0.935027)
    obj_feat.featureScaleMinMax("DE_RVI", 0.015280, 2.942026)
    obj_feat.featureScaleMinMax("DE_Beta", 0.552663, 0.998047)
    obj_feat.featureScaleMinMax("DE_m", 0.105326, 0.996094)
    obj_feat.featureScaleMinMax("DE_VH_asm", 0.000000, 0.364242)
    obj_feat.featureScaleMinMax("DE_VH_con", 0.000000, 74.302902)
    obj_feat.featureScaleMinMax("DE_VH_cor", -0.002264, 0.930006)
    obj_feat.featureScaleMinMax("DE_VH_dis", 0.000000, 6.512164)
    obj_feat.featureScaleMinMax("DE_VH_ent", 0.000000, 3.841390)
    obj_feat.featureScaleMinMax("DE_VH_hom", 0.000000, 0.677301)
    obj_feat.featureScaleMinMax("DE_VH_mean", 0.000000, 61.277348)
    obj_feat.featureScaleMinMax("DE_VH_var", 0.000000, 160.631638)
    obj_feat.featureScaleMinMax("DE_VV_asm", 0.000000, 0.439047)
    obj_feat.featureScaleMinMax("DE_VV_con", 0.000000, 92.837914)
    obj_feat.featureScaleMinMax("DE_VV_cor", -0.003553, 0.938837)
    obj_feat.featureScaleMinMax("DE_VV_dis", 0.000000, 7.227757)
    obj_feat.featureScaleMinMax("DE_VV_ent", 0.000000, 3.841390)
    obj_feat.featureScaleMinMax("DE_VV_hom", 0.000000, 0.721794)
    obj_feat.featureScaleMinMax("DE_VV_mean", 0.000000, 61.523441)
    obj_feat.featureScaleMinMax("DE_VV_var", 0.000000, 211.846954)
    obj_feat.featureScaleMinMax("AS_H", -0.6309294700622559, -0.00025027384981513023)
    obj_feat.featureScaleMinMax("AS_A", -0.9995785355567932, 0.9999527931213379)
    obj_feat.featureScaleMinMax("AS_Alpha", 0.0020001684315502644, 1.4092400074005127)
    obj_feat.featureScaleMinMax("DE_H", -0.6309216618537903, -9.606533421901986e-05)
    obj_feat.featureScaleMinMax("DE_A", -0.9984971284866333, 0.9999833703041077)
    obj_feat.featureScaleMinMax("DE_Alpha", 0.001635744352824986, 1.4519890546798706)


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

        self.feat_deal = qdFeatureDeal
        self.ssn_mod_dirname=self.init_dfn.fn(r"QingDao\SampleNumber\Mods")

    def shadowTraining(self):
        # spl_fn = self.sample_csv_spl_fn
        spl_fn = r"F:\ProjectSet\Shadow\Analysis\5\sh_qd_sample_spl2.csv"
        raster_fn = self.raster_fn
        model_dir = self.model_dfn.fn()
        model_name = self.model_name

        self.sct = ShadowCategoryTraining(model_dir, model_name, n_category=4,
                                          category_names=["IS", "VEG", "SOIL", "WAT"])
        self.sct.initCSVSample(spl_fn, ["IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"])
        self.sct.initSIC(raster_fn)
        qdFeatureDeal(self.sct)

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
    # sm_qd.sampleToCsv()
    # sm_qd.sampling()
    # sm_qd.shadowTraining()
    # sm_qd.testImdc()
    sm_qd.trainSampleNumber()
    pass


if __name__ == "__main__":
    # python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowMainQingDao import main; main()"
    main()
