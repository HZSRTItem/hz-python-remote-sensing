# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHIS.py
@Time    : 2024/4/10 10:52
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHIS
-----------------------------------------------------------------------------"""
import os

import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import random_split

from SRTCodes.GDALRasterIO import GDALRasterChannel
from SRTCodes.GDALUtils import GDALAccuracyImage
from SRTCodes.Utils import SRTLog, timeDirName, changefiledirname
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHFuncs import SHHReplaceCategoryImage, MLFuncs
from Shadow.Hierarchical.SHHRunMain import SHHModImPytorch, SHHModImSklearn
from Shadow.Hierarchical.ShadowHSample import SHH2Samples, SHH2_SPL, samplingSHH21OptSarGLCM


def convBnAct(
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        act=nn.ReLU()
):
    if act is None:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
        )
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        act,
    )


class ISModel_T(nn.Module):

    def __init__(self):
        super(ISModel_T, self).__init__()

        self.opt_cba1 = convBnAct(6, 16, 3, padding=1)
        self.opt_cba2 = convBnAct(16, 32, 3, padding=1)
        self.opt_max_pool1 = nn.MaxPool2d(2, 2)
        self.opt_cba3 = convBnAct(32, 64, 3, padding=1)
        self.opt_max_pool2 = nn.MaxPool2d(2, 2)
        self.opt_cba4 = convBnAct(64, 64, 3, padding=1, act=None)

        self.as_cba1 = convBnAct(2, 16, 3, padding=1)
        self.as_cba2 = convBnAct(16, 32, 3, padding=1)
        self.as_max_pool1 = nn.MaxPool2d(2, 2)
        self.as_cba3 = convBnAct(32, 64, 3, padding=1)
        self.as_max_pool2 = nn.MaxPool2d(2, 2)
        self.as_cba4 = convBnAct(64, 64, 3, padding=1, act=None)

        self.de_cba1 = convBnAct(2, 16, 3, padding=1)
        self.de_cba2 = convBnAct(16, 32, 3, padding=1)
        self.de_max_pool1 = nn.MaxPool2d(2, 2)
        self.de_cba3 = convBnAct(32, 64, 3, padding=1)
        self.de_max_pool2 = nn.MaxPool2d(2, 2)
        self.de_cba4 = convBnAct(64, 64, 3, padding=1, act=None)

        self.conv_cat1 = nn.AvgPool2d(2, stride=2)
        self.act1 = nn.Sigmoid()
        self.fc1 = nn.Linear(192, 2)

    def forward(self, x):
        x_opt = x[:, 6:12]
        x_opt = self.opt_cba1(x_opt)
        x_opt = self.opt_cba2(x_opt)
        x_opt = self.opt_max_pool1(x_opt)
        x_opt = self.opt_cba3(x_opt)
        x_opt = self.opt_max_pool2(x_opt)
        x_opt = self.opt_cba4(x_opt)

        x_as = x[:, 0:2]
        x_as = self.as_cba1(x_as)
        x_as = self.as_cba2(x_as)
        x_as = self.as_max_pool1(x_as)
        x_as = self.as_cba3(x_as)
        x_as = self.as_max_pool2(x_as)
        x_as = self.as_cba4(x_as)

        x_de = x[:, 3:5]
        x_de = self.de_cba1(x_de)
        x_de = self.de_cba2(x_de)
        x_de = self.de_max_pool1(x_de)
        x_de = self.de_cba3(x_de)
        x_de = self.de_max_pool2(x_de)
        x_de = self.de_cba4(x_de)

        x = torch.cat([x_opt, x_as, x_de], dim=1)
        x = self.conv_cat1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.act1(x)
        x = self.fc1(x)

        return x


def data_deal(x, y=None):
    x[0:2] = x[0:2] / 30 + 1
    x[3:5] = x[3:5] / 30 + 1
    x[6:12] = x[6:12] / 3000
    if y is not None:
        y = y - 1
        return x, y
    return x


def loadDS(win_size=None, is_random_split=False, is_test_field_name_split=True, mod_dirname=None, city_name="qd"):
    s2spl = loadS2SPL(win_size=win_size, city_name=city_name)

    if is_random_split:
        ds = s2spl.loadSHH2SamplesDataset(is_test=False, data_deal=data_deal)
        ds.ndc.__init__(3, win_size, (21, 21))
        train_ds, test_ds = random_split(dataset=ds, lengths=[0.8, 0.2], )
    elif is_test_field_name_split:
        train_ds, test_ds = s2spl.loadSHH2SamplesDataset(is_test=True, data_deal=data_deal)
        train_ds.ndc.__init__(3, win_size, (21, 21))
        test_ds.ndc.__init__(3, win_size, (21, 21))
    else:
        train_ds, test_ds = s2spl.loadSHH2SamplesDataset(is_test=False, data_deal=data_deal), None

    if mod_dirname is not None:
        s2spl.shh2_spl.toCSV(os.path.join(mod_dirname, "train_data.csv"), category_field_name="CATEGORY")

    print("length of train_ds:", len(train_ds))
    print("length of test_ds:", len(test_ds))
    return train_ds, test_ds


def loadSHH2Samples(win_size=None, city_name="qd"):
    s2spl = loadS2SPL(win_size, city_name=city_name)
    return s2spl.shh2_spl


def loadS2SPL(win_size=None, city_name="qd"):
    if city_name == "qd":
        map_dict = SHHConfig.CATE_MAP_IS_8
        s2spl = SHH2_SPL(map_dict=map_dict, others=0, is_npy=True)
        s2spl.add_shadow1samples(category_field_name="CATEGORY", map_dict=SHHConfig.CATE_MAP_IS_82)
        s2spl.filterCODEContain(1, 2)
        s2spl.filterEq("CITY", city_name)
        if win_size is not None:
            s2spl.shh2_spl.ndc.__init__(3, win_size, (21, 21))
        # s2spl.filterEq("IS_TAG", "TRUE")
        return s2spl
    elif city_name == "bj":
        map_dict = SHHConfig.CATE_MAP_IS_8
        s2spl = SHH2_SPL(map_dict=map_dict, others=0, is_npy=True)
        s2spl.add_shadow1samples(category_field_name="CATEGORY", map_dict=SHHConfig.CATE_MAP_IS_82)
        s2spl.filterCODEContain(1, 2)
        s2spl.filterEq("CITY", city_name)
        if win_size is not None:
            s2spl.shh2_spl.ndc.__init__(3, win_size, (21, 21))
        # s2spl.filterEq("IS_TAG", "TRUE")
        return s2spl
    elif city_name == "cd":
        map_dict = SHHConfig.CATE_MAP_IS_8
        s2spl = SHH2_SPL(map_dict=map_dict, others=0, is_npy=True)
        s2spl.add_shadow1samples(category_field_name="CATEGORY", map_dict=SHHConfig.CATE_MAP_IS_82)
        s2spl.filterCODEContain(1, 2)
        s2spl.filterEq("CITY", city_name)
        if win_size is not None:
            s2spl.shh2_spl.ndc.__init__(3, win_size, (21, 21))
        # s2spl.filterEq("IS_TAG", "TRUE")
        return s2spl


class ISMain:

    def __init__(self):
        self.smip = SHHModImPytorch()

        def func_predict(model, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_predict = func_predict

    def main(self, is_train=False, is_imdc=False):
        self.city_name = "cd"
        self.smip.model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\ISModels"
        self.smip.model_name = self.city_name + "_ISModel_T"
        self.smip.epochs = 100
        self.smip.device = "cuda:0"
        self.smip.n_test = 5
        self.smip.batch_size = 32
        self.smip.class_names = SHHConfig.SHH_CNAMES_VHL
        self.smip.n_class = len(self.smip.class_names)
        self.smip.win_size = (9, 9)
        self.smip.model = ISModel_T().to(self.smip.device)
        self.smip.func_predict = self.func_predict
        self.smip.func_y_deal = lambda y: y + 1
        self.smip.initISColorTable()

        # self.train()

        if is_train:
            self.train()
        if is_imdc:
            self.imdc()

        # self.predictSHH2SamplesToCSV(
        #     shh2_spl=loadSHH2Samples(self.smip.win_size),
        #     to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\13\sh2_spl13_1_modelpredict1.csv",
        #     mod_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240331H200249\VHLModel_VIT1_epoch70.pth",
        #     map_dict={2: 1, 1: 2, 3: 3}
        # )

    def train(self):
        self.smip.timeDirName()
        self.smip.train_ds, self.smip.test_ds = loadDS(
            self.smip.win_size, mod_dirname=self.smip.model_dirname, city_name=self.city_name)
        self.smip.train_ds.deal()
        self.smip.test_ds.deal()
        self.smip.initTrainLog()
        self.smip.initPytorchTraining()
        self.smip.pt.func_logit_category = self.func_predict
        self.smip.pt.func_y_deal = lambda y: y + 1
        self.smip.initModel()
        self.smip.initDataLoader()
        self.smip.initCriterion(nn.CrossEntropyLoss())
        self.smip.initOptimizer(torch.optim.Adam, lr=0.001, eps=10e-5)
        self.smip.pt.addScheduler(torch.optim.lr_scheduler.StepLR(self.smip.pt.optimizer, 20, gamma=0.5, last_epoch=-1))
        self.smip.copyFile(__file__)
        self.smip.print()
        # self.smip.pt.func_batch = lambda: print(len(x_id_list))
        self.smip.train()

    def imdc(self):
        self.smip.loadPTH(None)
        # self.smip.loadPTH(r"F:\ProjectSet\Shadow\Hierarchical\ISModels\20240410H115323\ISModel_T_epoch98.pth")
        if self.city_name == "qd":
            grc: GDALRasterChannel = GDALRasterChannel()
            grc.addGDALDatas(SHHConfig.SHH2_QD1_FNS[0])
            self.smip.imdc(grc=grc, is_jdt=True, data_deal=data_deal)
        elif self.city_name == "bj":
            self.smip.imdcTiles(
                tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1_retile",
                data_deal=data_deal,
            )
        elif self.city_name == "cd":
            self.smip.imdcTiles(
                tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1_retile",
                data_deal=data_deal,
            )

        # self.smip.imdcTiles(
        #     # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
        #     tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1_retile",
        #     data_deal=data_deal,
        # )
        pass

    def predictSHH2SamplesToCSV(self, shh2_spl: SHH2Samples, to_csv_fn, mod_fn=None,
                                to_category_field_name="NEW_CATEGORY", map_dict=None):
        print("to_csv_fn:", to_csv_fn)
        if mod_fn is not None:
            self.smip.loadPTH(mod_fn)
        shh2_spl = self.smip.predictSHH2Samples(
            shh2_spl, to_category_field_name=to_category_field_name, data_deal=data_deal)
        if map_dict is not None:
            shh2_spl.mapFiled(to_category_field_name, map_dict)
        shh2_spl.toCSV(to_csv_fn=to_csv_fn)


class ISMLMain:

    def __init__(self):
        self.fit_keys = None
        self.name = "_ISML"
        self.shh_mis = SHHModImSklearn()
        self.slog = SRTLog()
        self.model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\ISModels"

    def train(self, csv_fn=None, city_name="cd"):
        self.name = city_name.upper() + self.name
        self.model_dirname = timeDirName(self.model_dirname, is_mk=True)
        self.slog.__init__(os.path.join(self.model_dirname, "{0}_log.txt".format(self.name)), mode="a", )
        self.slog.kw("NAME", self.name)
        self.slog.kw("MODEL_DIRNAME", self.model_dirname)
        if csv_fn is not None:
            csv_fn = self.slog.kw("CSV_FN", csv_fn)
        else:
            # csv_fn = self.slog.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\19\shh2_spl19_1.csv")
            # csv_fn = self.slog.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shadow1samples\spl.csv")
            csv_fn = self.slog.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\21\sh2_spl21_sh2spl1.csv")
        self.shh_mis.initISColorTable()
        self.slog.kw("CATEGORY_NAMES", self.shh_mis.category_names)
        self.slog.kw("COLOR_TABLE", self.shh_mis.color_table)
        self.shh_mis.initPandas(pd.read_csv(csv_fn))
        self.shh_mis.dfFilterEQ("CITY", city_name)
        self.slog.kw("shh_mis.df.keys()", list(self.shh_mis.df.keys()))

        map_dict = {11: 2, 21: 0, 31: 1, 41: 0, 12: 0, 22: 0, 32: 0, 42: 0}
        # map_dict = None
        self.slog.kw("MAP DICT:", map_dict)
        self.slog.kw("Category Field Name:", self.shh_mis.initCategoryField(map_dict=map_dict))
        fit_keys = [
            'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
            'AS_VV', 'AS_VH', "AS_VVDVH",
            'DE_VV', 'DE_VH', "DE_VVDVH",
            "NDVI", "NDWI", "MNDWI",
            "OPT_asm", "OPT_con", "OPT_cor", "OPT_dis", "OPT_ent", "OPT_hom", "OPT_mean", "OPT_var",
            "AS_VH_asm", "AS_VH_con", "AS_VH_cor", "AS_VH_dis", "AS_VH_ent", "AS_VH_hom", "AS_VH_mean", "AS_VH_var",
            "AS_VV_asm", "AS_VV_con", "AS_VV_cor", "AS_VV_dis", "AS_VV_ent", "AS_VV_hom", "AS_VV_mean", "AS_VV_var",
            "DE_VH_asm", "DE_VH_con", "DE_VH_cor", "DE_VH_dis", "DE_VH_ent", "DE_VH_hom", "DE_VH_mean", "DE_VH_var",
            "DE_VV_asm", "DE_VV_con", "DE_VV_cor", "DE_VV_dis", "DE_VV_ent", "DE_VV_hom", "DE_VV_mean", "DE_VV_var",
        ]
        self.fit_keys = fit_keys

        self.slog.kw("fit_keys", fit_keys)
        self.shh_mis.addNDVIFeatExt()
        self.shh_mis.addNDWIFeatExt()
        self.shh_mis.addMNDWIFeatExt()
        self.shh_mis.addASVVDVH()
        self.shh_mis.addDEVVDVH()
        self.shh_mis.initXKeys(fit_keys)
        self.shh_mis.filterCategory(1, 2)
        self.shh_mis.testFieldSplitTrainTest()
        self.slog.kw("LEN X", len(self.shh_mis.x))
        self.slog.kw("LEN Train", len(self.shh_mis.x_train))
        self.slog.kw("LEN Test", len(self.shh_mis.y_test))
        self.shh_mis.initCLF(RandomForestClassifier(150))
        self.shh_mis.train()
        self.shh_mis.scoreTrainCM()
        self.slog.kw("Train CM", self.shh_mis.train_cm.fmtCM(), sep="\n")
        self.shh_mis.scoreTestCM()
        self.slog.kw("Test CM", self.shh_mis.test_cm.fmtCM(), sep="\n")
        mod_fn = self.slog.kw("Model FileName", os.path.join(self.model_dirname, "{0}.model".format(self.name)))
        self.shh_mis.saveModel(mod_fn)
        to_code_fn = self.slog.kw("to_code_fn", changefiledirname(__file__, self.model_dirname))
        self.shh_mis.saveCodeFile(code_fn=__file__, to_code_fn=to_code_fn)
        to_csv_fn = self.slog.kw("to_csv_fn", os.path.join(self.model_dirname, "{}_train_data.csv".format(self.name)))
        self.shh_mis.df.to_csv(to_csv_fn, index=False)
        to_imdc_fn = self.slog.kw("to_imdc_fn", os.path.join(self.model_dirname, "{}_imdc.tif".format(self.name)))

        if city_name == "qd":
            tiles_dirname = self.slog.kw(
                "TILES_DIRNAME", r"G:\ImageData\SHH2QingDaoImages\qd_sh2_1_opt_sar_glcm")
            self.shh_mis.imdcTiles(to_imdc_fn, tiles_dirname=tiles_dirname)
        elif city_name == "bj":
            tiles_dirname = self.slog.kw(
                "TILES_DIRNAME", r"G:\ImageData\SHH2BeiJingImages\bj_sh2_1_opt_sar_glcm")
            self.shh_mis.imdcTiles(to_imdc_fn, tiles_dirname=tiles_dirname)
        elif city_name == "cd":
            tiles_dirname = self.slog.kw(
                "TILES_DIRNAME", r"G:\ImageData\SHH2ChengDuImages\cd_sh2_1_opt_sar_glcm")
            self.shh_mis.imdcTiles(to_imdc_fn, tiles_dirname=tiles_dirname)

        return self.shh_mis.clf


def main():
    def func1():
        x = torch.rand((32, 12, 9, 9))
        model = ISModel_T()
        out_x = model(x)
        mod_fn = SHHConfig.tempFile(".pth")
        print(mod_fn)
        torch.save(model, mod_fn)

    def func2():
        # vhl_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240405H205015\QD_VHLML_imdc.tif",
        # vhl_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H195636\BJ_VHLML_imdc.tif"
        # vhl_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240413H195424\CD_VHLML_imdc.tif",
        # vhl_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H192733\VHLModel_VIT1_epoch90_imdc1.tif",
        SHHReplaceCategoryImage().VHL_IS(
            vhl_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240413H195424\CD_VHLML_imdc.tif",
            is_fn=r"F:\ProjectSet\Shadow\Hierarchical\ISModels\20240411H213328\cd_ISModel_T_epoch89_imdc1.tif",
            to_fn=r"F:\ProjectSet\Shadow\Hierarchical\ISModels\20240411H213328\cd_ISModel_T_epoch89_imdc1_vhl.tif",
        )

    def func_sampling():
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shadow1samples\spl.csv"
        to_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\21\sh2_spl21_sh2spl1.csv"

        samplingSHH21OptSarGLCM(csv_fn, to_csv_fn)

        df = pd.read_csv(to_csv_fn)

        def norm(name1, name2):
            return (df[name1] - df[name2]) / (df[name1] + df[name2])

        df["NDVI"] = norm("B8", "B4")
        df["NDWI"] = norm("B3", "B8")
        df["MNDWI"] = norm("B3", "B12")
        df["AS_VHDVV"] = df["AS_VH"] - df["AS_VV"]
        df["DE_VHDVV"] = df["DE_VH"] - df["DE_VV"]

        df.to_csv(to_csv_fn, index=False)

    def func3():
        MLFuncs().fit1Upate(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\20\sh2_spl20_qdis21.csv",
            to_csv_fn=None, is_tag_field_name="IS_TAG", category_field_name="CATEGORY_CODE",
            clf=None, update_field_name="CATEGORY"
        )

    def func4():
        cm = GDALAccuracyImage(
            # imdc_geo_fn=r"F:\ProjectSet\Shadow\Hierarchical\ISModels\20240413H171853\QD_ISML_imdc_vhl.tif",
            imdc_geo_fn=r"F:\ProjectSet\Shadow\Hierarchical\ISModels\20240411H204838\ISModel_T_epoch90_imdc1_vhl.tif",
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\20\sh2_spl20_qdis21.csv",
            category_field_name="CATEGORY_CODE",
            x_field_name="X",
            y_field_name="Y",
            cnames=["SOIL", "IS"],
            imdc_map_dict={1: 2, 3: 1},
            df_map_dict=None,
        ).fit()
        print(cm.fmtCM())

    func2()
    # ISMain().main(is_train=True)
    # ISMLMain().train()

    pass


if __name__ == "__main__":
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHIS import ISMain; ISMain().main(is_imdc=True)"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHIS import ISMain; ISMain().main(is_train=True)"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHIS import ISMLMain; ISMLMain().train()"

    """
    main()
