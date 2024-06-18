# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowMain3.py
@Time    : 2024/5/10 14:46
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowMain3
-----------------------------------------------------------------------------"""
import csv
import os
from shutil import copyfile

import numpy as np
import pandas as pd

from SRTCodes.GDALRasterClassification import GDALImdcAcc
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALSamplingFast
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.PandasUtils import splitDf
from SRTCodes.SRTReadWrite import SRTInfoFileRW
from SRTCodes.Utils import DirFileName, Jdt, filterFileContain, getfilenamewithoutext, numberfilename, readText
from Shadow.ShadowMain import ShadowMain
from Shadow.ShadowMainBeiJing import bjFeatureDeal
from Shadow.ShadowMainChengDu import cdFeatureDeal
from Shadow.ShadowMainQingDao import qdFeatureDeal
from Shadow.ShadowTraining import ShadowCategoryTraining
from Shadow.ShadowUtils import ShadowTiaoTestAcc


class ShadowCategoryTrainingFeatures(ShadowCategoryTraining):

    def __init__(self, model_dir, model_name, n_category=2, category_names=None):
        super().__init__(model_dir, model_name, n_category, category_names)

    def getFeature(self):
        if self._n_iter == len(self._feat_iter):
            self._n_iter = 0
            return False
        self._feat_types = [self._feat_iter[self._n_iter]]
        self._feats = []
        for feat_type in self._feat_types:
            if feat_type == self.front_feat_type_name:
                continue
            self._feats.extend(self.feat_types[feat_type])
        self._n_iter += 1
        return True

    def _initFeatures(self):
        self._feat_iter = list(self.feat_types.keys())
        return


class ShadowMain3BJ(ShadowMain):

    def __init__(self, city_name="bj"):
        super().__init__()
        self.city_name = city_name
        if city_name == "bj":
            self.model_name = "BeiJing"

            self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingImages")
            self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingSamples")
            self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\BeiJing\Mods")

            self.raster_fn = self.raster_dfn.fn("SH_BJ_envi.dat")
            self.sample_fn = self.sample_dfn.fn("BeiJingSamples.xlsx")
            self.sample_csv_fn = self.sample_dfn.fn("sh_bj_sample.csv")
            self.sample_csv_spl_fn = self.sample_dfn.fn("sh_bj_sample_spl.csv")

            self.feat_deal = bjFeatureDeal
            self.ssn_mod_dirname = self.init_dfn.fn(r"BeiJing\SampleNumber\Mods")
        elif city_name == "qd":
            self.model_name = "QingDao"

            self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\QingDaoImages")
            self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\QingDaoSamples")
            self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\QingDao\Mods")

            self.raster_fn = self.raster_dfn.fn("SH_QD_envi.dat")
            self.sample_fn = self.sample_dfn.fn("QingDaoSamples.xlsx")
            self.sample_csv_fn = self.sample_dfn.fn("sh_qd_sample.csv")
            self.sample_csv_spl_fn = self.sample_dfn.fn("sh_qd_sample_spl.csv")

            self.feat_deal = qdFeatureDeal
            self.ssn_mod_dirname = self.init_dfn.fn(r"QingDao\SampleNumber\Mods")
        elif city_name == "cd":
            self.model_name = "ChengDu"

            self.raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\ChengDuImages")
            self.sample_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\ChengDuSamples")
            self.model_dfn = DirFileName(r"F:\ProjectSet\Shadow\ChengDu\Mods")

            self.raster_fn = self.raster_dfn.fn("SH_CD_envi.dat")
            self.sample_fn = self.sample_dfn.fn("ChengDuSamples.xlsx")
            self.sample_csv_fn = self.sample_dfn.fn("sh_cd_sample.csv")
            self.sample_csv_spl_fn = self.sample_dfn.fn("sh_cd_sample_spl.csv")

            self.feat_deal = cdFeatureDeal
            self.ssn_mod_dirname = self.init_dfn.fn(r"ChengDu\SampleNumber\Mods")

    def shadowTraining(self):
        # spl_fn = self.sample_csv_spl_fn
        # spl_fn = r"F:\ProjectSet\Shadow\Analysis\5\sh_bj_sample_spl2.csv"
        raster_fn, spl_fn = self.initSampleRasterFN()
        init_dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")
        spl_fn = init_dfn.fn(self.city_name.upper(), "SH_{}_spl.csv".format(self.city_name.upper()))

        model_dir = self.model_dfn.fn()
        model_name = self.model_name

        # ShadowCategoryTrainingFeatures ShadowCategoryTraining
        self.sct = ShadowCategoryTrainingFeatures(model_dir, model_name, n_category=4,
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
        # self.sctAddFeatureType("optics_as_de")

        opt_feats = (
            "Blue", "Green", "Red", "NIR", "NDVI", "NDWI",
            # "OPT_dis", "OPT_hom", "OPT_mean", "OPT_var"
        )
        self.sct.addFeatureType("OPT", *opt_feats)
        self.sct.addFeatureType("OPT-BSC", *opt_feats,
                                "AS_VV", "AS_VH", "AS_VHDVV", "DE_VV", "DE_VH", "DE_VHDVV", )
        self.sct.addFeatureType("OPT-C2", *opt_feats,
                                "AS_C11", "AS_C22", "AS_Lambda1", "AS_Lambda2", "AS_SPAN",
                                "DE_C11", "DE_C22", "DE_Lambda1", "DE_Lambda2", "DE_SPAN", )
        self.sct.addFeatureType("OPT-HA", *opt_feats,
                                "AS_H", "AS_A", "AS_Alpha", "DE_H", "DE_A", "DE_Alpha", )
        self.sct.addFeatureType("OPT-GLCM", *opt_feats,
                                "AS_VH_hom", "AS_VH_mean", "AS_VH_var", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
                                "DE_VH_hom", "DE_VH_mean", "DE_VH_var", "DE_VV_hom", "DE_VV_mean", "DE_VV_var", )

        #   "Select1200ToTest", "Select", "Select600ToTest", "NDWIRANDOM", "NDWIWAT",
        if self.city_name == "bj":
            self.sct.addTagType("TAG", "Select1200ToTest", "Select", "Select600ToTest", "NDWIRANDOM", "NDWIWAT", )
        elif self.city_name == "qd":
            self.sct.addTagType("TAG", "SELECT", "STATIFY_SH", "GE_MAP_RAND", "SEA", "NIR_SH", "STATIFY_NOSH", )
        elif self.city_name == "cd":
            self.sct.addTagType("TAG", "Random3000", "NDWIRandom", "Select")

        self.sct.print()

        to_code_fn = os.path.join(self.sct.model_dir, os.path.split(__file__)[1])
        print(to_code_fn)
        copyfile(__file__, to_code_fn)

        self.sct.train()

    def initSampleRasterFN(self):
        if self.city_name == "bj":
            spl_fn = r"F:\ProjectSet\Shadow\Analysis\8\test\bj_train_data_t2_spl.csv"
        elif self.city_name == "qd":
            spl_fn = r"F:\ProjectSet\Shadow\Analysis\5\sh_qd_sample_spl2.csv"
        elif self.city_name == "cd":
            spl_fn = r"F:\ProjectSet\Shadow\ChengDu\Samples\3\train_data_cd.csv"
        else:
            spl_fn = ""
        init_dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")
        raster_fn = init_dfn.fn(self.city_name.upper(), "SH_{}_envi.dat".format(self.city_name.upper()))
        print("SPL_FN:", spl_fn)
        print("RASTER_FN:", raster_fn)
        return raster_fn, spl_fn

    def sampling_3(self):
        raster_fn, spl_fn = self.initSampleRasterFN()
        init_dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")
        to_csv_fn = init_dfn.fn(self.city_name.upper(), "SH_{}_spl.csv".format(self.city_name.upper()))
        print("TO_CSV_FN:", to_csv_fn)
        # self.sampling( csv_fn=spl_fn, gr=GDALRaster(raster_fn), to_csv_fn=to_csv_fn)
        grf = GDALSamplingFast(raster_fn)
        grf.csvfile(spl_fn, to_csv_fn)


def featExtHA():
    init_dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")

    # raster_dfn = DirFileName(r"F:\ProjectSet\Shadow\Release\BeiJingImages")
    # raster_fn = raster_dfn.fn("SH_BJ_envi.dat")

    def func3(city_name, raster_fn):

        gr = GDALRaster(raster_fn)

        def func1(name, dfn):
            c11_key = "{}_C11".format(name)
            c22_key = "{}_C22".format(name)
            c12_real_key = "{}_C12_real".format(name)
            c12_imag_key = "{}_C12_imag".format(name)

            d_c11 = gr.readGDALBand(c11_key)
            d_c22 = gr.readGDALBand(c22_key)
            d_c12_real = gr.readGDALBand(c12_real_key)
            d_c12_imag = gr.readGDALBand(c12_imag_key)

            lamd1, lamd2 = np.zeros((gr.n_rows, gr.n_columns)), np.zeros((gr.n_rows, gr.n_columns))
            alp1, alp2 = np.zeros((gr.n_rows, gr.n_columns)), np.zeros((gr.n_rows, gr.n_columns))

            jdt = Jdt(gr.n_rows, "{0} {1} featExtHA".format(city_name, name)).start()
            for i in range(gr.n_rows):
                for j in range(gr.n_columns):
                    c2 = np.array([
                        [d_c11[i, j], d_c12_real[i, j] + (d_c12_imag[i, j] * 1j)],
                        [d_c12_real[i, j] - (d_c12_imag[i, j] * 1j), d_c22[i, j]],
                    ])
                    eigenvalue, featurevector = np.linalg.eig(c2)
                    lamd1[i, j] = np.abs(eigenvalue[0])
                    lamd2[i, j] = np.abs(eigenvalue[1])
                    alp1[i, j] = np.arccos(abs(featurevector[0, 0]))
                    alp2[i, j] = np.arccos(abs(featurevector[0, 1]))
                jdt.add()
            jdt.end()

            # dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")
            # gr.save(lamd1, dfn.fn("lamd1.dat"))
            # gr.save(lamd2, dfn.fn("lamd2.dat"))
            # gr.save(alp1, dfn.fn("alp1.dat"))
            # gr.save(alp2, dfn.fn("alp2.dat"))

            p1 = lamd1 / (lamd1 + lamd2)
            p2 = lamd2 / (lamd1 + lamd2)
            d_h = p1 * (np.log(p1) / np.log(3)) + p2 * (np.log(p2) / np.log(3))
            a = p1 - p2
            alp = p1 * alp1 + p2 * alp2

            gr.save(d_h, dfn.fn("{}_H.dat".format(name)), descriptions=["{}_H".format(name)])
            gr.save(a, dfn.fn("{}_A.dat".format(name)), descriptions=["{}_A".format(name)])
            gr.save(alp, dfn.fn("{}_Alpha.dat".format(name)), descriptions=["{}_Alpha".format(name)])

        func1("AS", DirFileName(init_dfn.fn(city_name)))
        func1("DE", DirFileName(init_dfn.fn(city_name)))

    # func3("BJ", r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat")
    # func3("CD", r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat")
    # func3("QD", r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat")

    def func4(city_name, raster_fn):
        print(city_name, raster_fn)
        gr = GDALRaster(raster_fn)
        names = gr.names.copy()
        data_list = [gr.readAsArray()]

        def func1(name, dfn):
            names.extend(["{}_H".format(name), "{}_A".format(name), "{}_Alpha".format(name)])
            data_list.extend([
                [GDALRaster(dfn.fn("{}_H.dat".format(name))).readAsArray()],
                [GDALRaster(dfn.fn("{}_A.dat".format(name))).readAsArray()],
                [GDALRaster(dfn.fn("{}_Alpha.dat".format(name))).readAsArray()]
            ])

        func1("AS", DirFileName(init_dfn.fn(city_name)))
        func1("DE", DirFileName(init_dfn.fn(city_name)))
        data = np.concatenate(data_list)
        to_fn = DirFileName(init_dfn.fn(city_name)).fn(os.path.split(raster_fn)[1])
        print(to_fn)
        gr.save(data, to_fn, descriptions=names)

    # func4("BJ", r"F:\ProjectSet\Shadow\Release\BeiJingImages\SH_BJ_envi.dat")
    # func4("CD", r"F:\ProjectSet\Shadow\Release\ChengDuImages\SH_CD_envi.dat")
    # func4("QD", r"F:\ProjectSet\Shadow\Release\QingDaoImages\SH_QD_envi.dat")

    def func5(city_name, filename):
        gr = GDALRaster(init_dfn.fn(city_name, filename))
        print(city_name)
        names = ["AS_H", "AS_A", "AS_Alpha", "DE_H", "DE_A", "DE_Alpha", ]
        for name in names:
            data = gr.readGDALBand(name)
            print("obj_feat.featureScaleMinMax(\"{0}\", {1}, {2})".format(name, data.min(), data.max()))

            # y, x = np.histogram(data, bins=256)
            # plt.plot(x[:-1], y)
            # plt.title(name)
            # plt.show()

    func5("BJ", "SH_BJ_envi.dat")
    func5("QD", "SH_QD_envi.dat")
    func5("CD", "SH_CD_envi.dat")

    def func2():
        gr = GDALRaster()
        c11_key = "AS_C11"
        c22_key = "AS_C22"
        c12_real_key = "AS_C12_real"
        c12_imag_key = "AS_C12_imag"

        d_c11 = gr.readGDALBand(c11_key)
        d_c22 = gr.readGDALBand(c22_key)
        d_c12_real = gr.readGDALBand(c12_real_key)
        d_c12_imag = gr.readGDALBand(c12_imag_key)
        lamd1_tmp = gr.readGDALBand("AS_Lambda1")
        lamd2_tmp = gr.readGDALBand("AS_Lambda2")
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Analysis\14")
        lamd1 = GDALRaster(dfn.fn("lamd1.dat")).readGDALBand(1)
        lamd2 = GDALRaster(dfn.fn("lamd2.dat")).readGDALBand(1)
        alp1 = GDALRaster(dfn.fn("alp1.dat")).readGDALBand(1)
        alp2 = GDALRaster(dfn.fn("alp2.dat")).readGDALBand(1)

        p1 = lamd1 / (lamd1 + lamd2)
        p2 = lamd2 / (lamd1 + lamd2)
        d_h = p1 * (np.log(p1) / np.log(3)) + p2 * (np.log(p2) / np.log(3))
        alp = p1 * alp1 + p2 * alp2

        gr.save(d_h, dfn.fn("h.dat"))
        gr.save(alp, dfn.fn("alp.dat"))



def accuracyImdc():

    # QingDao
    # dirname = r"F:\ProjectSet\Shadow\QingDao\Mods\20240510H202956"
    # dirname = r"F:\ProjectSet\Shadow\QingDao\Mods\20240510H224639"

    # BeiJing
    # dirname = r"F:\ProjectSet\Shadow\BeiJing\Mods\20240510H214314"
    # dirname = r"F:\ProjectSet\Shadow\BeiJing\Mods\20240510H235533"

    # ChengDu
    # dirname = r"F:\ProjectSet\Shadow\ChengDu\Mods\20240510H231256"
    dirname = r"F:\ProjectSet\Shadow\ChengDu\Mods\20240511H012421"

    dirname_list = [
        r"F:\ProjectSet\Shadow\ChengDu\Mods\20240222H170152",
        r"F:\ProjectSet\Shadow\QingDao\Mods\20231226H093225",
        r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303",
    ]

    # dirname = r"F:\ProjectSet\Shadow\BeiJing\Mods\20231225H110303"
    # csv_fn = r"F:\ProjectSet\Shadow\Analysis\14\samples\bj_qjy_2.csv"
    csv_fn = r"F:\ProjectSet\Shadow\Analysis\14\samples\cd_qjy_3.csv"
    # csv_fn = r"F:\ProjectSet\Shadow\Analysis\14\samples\qd_qjy_1.csv"

    fns = filterFileContain(dirname, "_imdc.dat")
    fns = [os.path.join(dirname, fn) for fn in os.listdir(dirname) if fn.endswith("_imdc.dat")]
    fns = [fn for fn in fns if "SPL_SH-SVM-TAG" in fn]
    cnames = ["IS", "VEG", "SOIL", "WAT"]
    to_dict = {
        "NAME": [],
        "OA": [], "Kappa": [],
        **{"{} OA".format(cname): [] for cname in cnames},
        **{"{} Kappa".format(cname): [] for cname in cnames},
        **{"{} UA".format(cname): [] for cname in cnames},
        **{"{} PA".format(cname): [] for cname in cnames},
    }

    def map_df(df):
        map_dict = {"NOT_KNOW": 0, "IS": 11, "VEG": 21, "SOIL": 31, "WAT": 41, "IS_SH": 12, "VEG_SH": 22, "SOIL_SH": 32,
                    "WAT_SH": 42}
        category_list = []
        for i in range(len(df)):
            cname = str(df["CATEGORY_NAME"][i])
            category_list.append(map_dict[cname])
        df["CATEGORY"] = category_list
        # df["X"] = df["__X"]
        # df["Y"] = df["__Y"]
        return df

    def imdc_acc(imdc_fn):
        to_dict["NAME"].append(getfilenamewithoutext(imdc_fn))
        gica = GDALImdcAcc(imdc_fn)
        df = pd.read_csv(csv_fn)
        # df = df[df["TEST"] == 0]
        df = map_df(df)
        gica.addDataFrame(df)
        gica.map_category = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
        gica.calCM(cnames)
        to_dict["OA"].append(gica.cm.OA())
        to_dict["Kappa"].append(gica.cm.getKappa())
        print(gica.cm.fmtCM())
        for name in gica.cm.CNAMES():
            cm = gica.cm.accuracyCategory(name)
            print(name)
            to_dict["{} OA".format(name)].append(cm.OA())
            to_dict["{} Kappa".format(name)].append(cm.getKappa())
            to_dict["{} UA".format(name)].append(cm.UA()[0])
            to_dict["{} PA".format(name)].append(cm.PA()[0])
            print(cm.fmtCM())
        return gica.cm

    for fn in fns:
        imdc_acc(fn)
    to_df = pd.DataFrame(to_dict)
    pd.set_option('display.max_columns', 1000)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 1000)
    print(to_df.sort_values("IS OA", ascending=False)[["NAME", "OA", "IS OA", "IS Kappa"]])
    to_csv_fn = numberfilename(r"F:\ProjectSet\Shadow\Analysis\14\test\imdc_test.csv")
    print(to_csv_fn)
    to_df.to_csv(to_csv_fn)


def calOA():
    cm_fn_dict1 = {
        "qd": r"F:\ProjectSet\Shadow\Analysis\11\3\qdcm.txt",
        "bj": r"F:\ProjectSet\Shadow\Analysis\11\3\bjcm.txt",
        "cd": r"F:\ProjectSet\Shadow\Analysis\11\3\cdcm.txt",
    }

    cm_fn_dict2 = {
        "qd": r"F:\ProjectSet\Shadow\Analysis\11\3\cm_nosh_qd.txt",
        "bj": r"F:\ProjectSet\Shadow\Analysis\11\3\cm_nosh_bj.txt",
        "cd": r"F:\ProjectSet\Shadow\Analysis\11\3\cm_nosh_cd.txt",
    }

    cm_fn_dict3 = {
        "qd": r"F:\ProjectSet\Shadow\Analysis\11\3\spl_nosh_cm_sh_qd.txt",
        "bj": r"F:\ProjectSet\Shadow\Analysis\11\3\spl_nosh_cm_sh_bj.txt",
        "cd": r"F:\ProjectSet\Shadow\Analysis\11\3\spl_nosh_cm_sh_cd.txt",
    }

    cm_fn_dict4 = {
        "qd": r"F:\ProjectSet\Shadow\Analysis\11\3\spl_nosh_cm_nosh_qd.txt",
        "bj": r"F:\ProjectSet\Shadow\Analysis\11\3\spl_nosh_cm_nosh_bj.txt",
        "cd": r"F:\ProjectSet\Shadow\Analysis\11\3\spl_nosh_cm_nosh_cd.txt",
    }

    cm_fn_dict_list = [cm_fn_dict1, cm_fn_dict2, cm_fn_dict3, cm_fn_dict4]
    df_list = []
    fn_list = []

    def cm_oa_kappa_dict(cm_fn_dict):
        cm_lines = []

        def cm_oa_kappa(spl_fn, city_name):
            sif_rw = SRTInfoFileRW(spl_fn)
            to_dict = sif_rw.readAsDict()
            print(to_dict)
            to_dict_oa = {"City": city_name, "standard": "OA"}
            to_dict_kappa = {"City": city_name, "standard": "Kappa"}
            for k in to_dict:
                print(k)
                line_list = []
                for line in to_dict[k][1:5]:
                    line = line.strip()
                    lines = line.split(" ")
                    lines = lines[1:5]
                    line_list.append(list(map(float, lines)))
                    print(" ".join(lines))
                data = np.array(line_list)

                cm_category = np.zeros((2, 2))
                cm_category[0, 0] = int(data[0, 0])
                cm_category[0, 1] = int(np.sum(data[0, 1:]))
                cm_category[1, 0] = int(np.sum(data[1:, 0]))
                cm_category[1, 1] = int(np.sum(data[1:, 1:]))

                cm = ConfusionMatrix(2, ["IS", "NOT_KNOW"])
                cm._cm = cm_category
                cm._cm_accuracy = cm.calCM()

                d1 = (data[0, 0] + np.sum(data[1:, 1:]))
                d2 = np.sum(data)
                print(d1 / d2)
                print(cm.fmtCM())
                print(cm.getKappa())

                to_dict_oa[k] = "{:.2f}%".format(cm.OA())
                to_dict_kappa[k] = "{:.4f}".format(cm.getKappa())

                # to_dict_oa[k] = cm.OA()
                # to_dict_kappa[k] = cm.getKappa()

            return [to_dict_oa, to_dict_kappa]

        for k in cm_fn_dict:
            cm_lines.extend(cm_oa_kappa(cm_fn_dict[k], k))

        pd.set_option('display.max_columns', 1000)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)
        to_df = pd.DataFrame(cm_lines)
        print(pd.DataFrame(cm_lines))
        to_fn = numberfilename(r"F:\ProjectSet\Shadow\Analysis\11\3\cm_t.csv")
        pd.DataFrame(cm_lines).to_csv(to_fn, index=False)

        df_list.append(to_df)
        fn_list.append(to_fn)

    for d in cm_fn_dict_list:
        cm_oa_kappa_dict(d)

    with open(numberfilename(r"F:\ProjectSet\Shadow\Analysis\11\3\cm_t.csv"), "w", encoding="utf-8") as f:
        for fn in fn_list:
            f.write(readText(fn))
            f.write("\n")


class STTA1(ShadowTiaoTestAcc):

    def __init__(self, init_dirname=None, init_name=None):
        super().__init__(init_dirname, init_name)

    def acc1(self, name, is_save=False):
        to_dirname = os.path.join(self.init_dirname, "QJY_{0}".format(name))

        csv_fn = os.path.join(to_dirname, "{0}_qjy_{1}.csv".format(self.init_name, name))
        txt_fn = os.path.join(to_dirname, "{0}_qjy_{1}.txt".format(self.init_name, name))

        srt_fr = SRTInfoFileRW(txt_fn)
        d = srt_fr.readAsDict()
        df_dict = {"__X": [], "__Y": [], "__CNAME": [], "__IS_TAG": []}
        for k in d["FIELDS"]:
            df_dict[k] = []

        fields = d["FIELDS"].copy()
        cr = csv.reader(d["DATA"])
        for line in cr:
            for k in df_dict:
                df_dict[k].append(None)

            x = float(line[3])
            y = float(line[4])
            c_name = line[1].strip()
            is_tag = eval(line[2])
            df_dict["__X"][-1] = x
            df_dict["__Y"][-1] = y
            df_dict["__CNAME"][-1] = c_name
            df_dict["__IS_TAG"][-1] = is_tag

            for i in range(5, len(line)):
                df_dict[fields[i - 5]][-1] = line[i]

        df_dict = self.samplingNew(df_dict, csv_fn)

        c_list = [self.category_map[cname] for cname in df_dict["__CNAME"]]

        df_acc = []
        cm_dict = {}
        for k in self.imdc_fns:
            print(k)
            imdc_list = _listToInt(df_dict[k])
            cm = ConfusionMatrix(4, self.cm_names)
            cm.addData(c_list, imdc_list)
            print(cm.fmtCM())
            cm_dict[k] = cm.fmtCM()
            to_dict = {"NAME": k, "OA": cm.OA(), "Kappa": cm.getKappa(), }
            for cm_name in cm:
                to_dict[cm_name + " UATest"] = cm.UA(cm_name)
                to_dict[cm_name + " PATest"] = cm.PA(cm_name)
            df_acc.append(to_dict)

        df_acc = pd.DataFrame(df_acc)
        df_acc = df_acc.sort_values("OA", ascending=False)
        pd.options.display.precision = 2
        print(df_acc)
        pd.options.display.precision = 6

        if is_save:
            to_dirname_save = timeDirName(to_dirname, True)
            time_name = os.path.split(to_dirname_save)[1]

            to_csv_fn = changefiledirname(csv_fn, to_dirname_save)
            to_csv_fn = changext(to_csv_fn, "_{0}.csv".format(time_name))
            shutil.copyfile(csv_fn, to_csv_fn)

            to_txt_fn = changefiledirname(txt_fn, to_dirname_save)
            to_txt_fn = changext(to_txt_fn, "_{0}.txt".format(time_name))
            shutil.copyfile(csv_fn, to_txt_fn)

            df_acc.to_csv(os.path.join(to_dirname_save, "{0}_acc.csv".format(time_name)))
            to_cm_fn = os.path.join(to_dirname_save, "{0}_cm.txt".format(time_name))
            with open(to_cm_fn, "w", encoding="utf-8") as fw:
                for k, cm in cm_dict.items():
                    print(">", k, file=fw)
                    print(cm, file=fw)
                    print(file=fw)

        return


def accTiao1():

    # txt to csv
    stta = ShadowTiaoTestAcc(r"F:\ProjectSet\Shadow\Analysis\11\cd")
    # stta.buildNew(r"F:\ProjectSet\Shadow\ChengDu\Mods\20231226H093253")
    # stta.buildNew(r"F:\ProjectSet\Shadow\ChengDu\Mods\20240222H170152")
    stta.load()
    # stta.categoryMap(NOT_KNOW=0, IS=1, VEG=2, SOIL=3, WAT=4, IS_SH=1, VEG_SH=2, SOIL_SH=3, WAT_SH=4)
    # stta.cm_names = ["IS", "VEG", "SOIL", "WAT"]
    # stta.updateTrueFalseColumn(True)
    # stta.sumColumns("SUM1", "TF_", "OPTICS")
    # stta.sortColumn(["SUM1", "CATEGORY"], ascending=True)
    # stta.addQJY("3", field_names=[
    #     'SRT', 'X', 'Y', 'CNAME', 'CATEGORY', 'TAG', 'TEST', 'NDVI', 'NDWI', 'SUM1'
    # ], TEST=0)
    # stta.accQJY("3", is_save=True)
    stta.accSHQJY("3", is_save=True)
    stta.saveDFToCSV()
    stta.save()


def main():
    df = pd.read_csv(r"F:\ProjectSet\Shadow\Analysis\14\samples\bj\bj_spl1.csv")
    print(df["CATEGORY_NAME"].value_counts())
    print(df.keys())
    df_is, df_shis, df_nois = splitDf(df, "CATEGORY_NAME", ["IS"], [ "IS_SH"], ["VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH", ])
    print("len df_nois", len(df_is))
    print("len df_nois", len(df_nois))
    df_nois_spl = df_nois.sample(len(df_is) + 50)
    print(pd.unique(df_nois_spl["CATEGORY_NAME"]))
    to_df = pd.concat([df_is, df_nois_spl, df_shis.sample(50)])
    print(to_df["CATEGORY_NAME"].value_counts())
    to_df.to_csv(r"F:\ProjectSet\Shadow\Analysis\14\samples\bj\bj_spl2.csv")

    pass


def method_name1():
    sm = ShadowMain3BJ()
    sm.shadowTraining()


if __name__ == "__main__":
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowMain3 import ShadowMain3BJ; ShadowMain3BJ().shadowTraining()" 
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.ShadowMain3 import featExtHA; featExtHA()" 
    """
    accuracyImdc()
