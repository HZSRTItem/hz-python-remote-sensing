# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHVHL.py
@Time    : 2024/3/31 10:40
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHVHLModelDataset
-----------------------------------------------------------------------------"""
import csv
import os.path
from functools import partial

import pandas as pd
import torch
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier
from torch import nn
from torch.utils.data import random_split

from RUN.RUNFucs import splTxt2Dict
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALAccuracyConfusionMatrix
from SRTCodes.NumpyUtils import categoryMap
from SRTCodes.PandasUtils import DataFrameDictSort
from SRTCodes.SRTReadWrite import SRTInfoFileRW
from SRTCodes.Utils import numberfilename, readJson, SRTLog, timeDirName, changefiledirname, concatCSV, savecsv
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHRunMain import SHHModImPytorch, SHHModImSklearn
from Shadow.Hierarchical.SHHTransformer import VisionTransformer_Channel
from Shadow.Hierarchical.ShadowHSample import SHH2_SPL, SHH2Samples, samplingSHH21OptSarGLCM


class VHLModel_VIT1(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vit_c = VisionTransformer_Channel(
            in_channels=6,
            image_size=8,
            patch_size=2,
            num_layers=12,
            num_heads=12,
            hidden_dim=120,
            mlp_dim=600,
            dropout=0.2,
            attention_dropout=0.2,
            num_classes=3,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs=None,
        )

    def forward(self, x):
        x = self.vit_c(x)
        return x


class VHLModel_Net2(nn.Module):

    def __init__(self):
        super(VHLModel_Net2, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(1152, 3),
            # nn.ReLU(),
            # nn.Linear(256, 3),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


def data_deal(x, y=None):
    # x[0:2] = x[0:2] / 30 + 1
    # x[2:4] = x[3:5] / 30 + 1
    # x[4:10] = x[6:] / 3000
    # x = x[:10]
    x = x[6:] / 3000
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


def loadSHH2Samples(win_size=None):
    s2spl = loadS2SPL(win_size)
    return s2spl.shh2_spl


def loadS2SPL(win_size=None, city_name="qd"):
    if city_name == "qd":
        map_dict = {1: 2, 2: 1, 3: 3}
        s2spl = SHH2_SPL(map_dict=map_dict, others=0, is_npy=True)
        s2spl.add_qd_VHL_random2000(category_field_name="CATEGORY_CODE")
        map_dict = {1: 1, 2: 2, 3: 3}
        s2spl.add_qd_VHL_random10000(category_field_name="CATEGORY", map_dict=map_dict)
        map_dict = {11: 2}
        s2spl.add_qd_roads_shouhua_tp1(category_field_name="CATEGORY", map_dict=map_dict)
        if win_size is not None:
            s2spl.shh2_spl.ndc.__init__(3, win_size, (21, 21))
        # s2spl.filterEq("IS_TAG", "TRUE")
        return s2spl
    elif city_name == "bj":
        map_dict = {1: 1, 2: 2, 3: 3}
        s2spl = SHH2_SPL(map_dict=map_dict, others=0, is_npy=True)
        s2spl.add_bj_vhl_random2000(is_npy=True)
        s2spl.add_bj_vhl_random10000(is_npy=True, selects={1: 1000, 2: 600, 3: None}, field_datas={"TEST": 1})
        if win_size is not None:
            s2spl.shh2_spl.ndc.__init__(3, win_size, (21, 21))
        # s2spl.filterEq("IS_TAG", "TRUE")
        return s2spl
    elif city_name == "cd":
        map_dict = {1: 1, 2: 2, 3: 3}
        s2spl = SHH2_SPL(map_dict=map_dict, others=0, is_npy=True)
        s2spl.add_cd_vhl_random2000(is_npy=True)
        # s2spl.add_bj_vhl_random10000(is_npy=True, selects={1: 1000, 2: 600, 3: None}, field_datas={"TEST": 1})
        if win_size is not None:
            s2spl.shh2_spl.ndc.__init__(3, win_size, (21, 21))
        # s2spl.filterEq("IS_TAG", "TRUE")
        return s2spl


class VHLSamplesMain:

    def __init__(self):
        self.name = "name"


class VHLMLMain:

    def __init__(self):
        self.fit_keys = None
        self.name = "_VHLML"
        self.shh_mis = SHHModImSklearn()
        self.slog = SRTLog()
        self.model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\VHLModels"

    def train(self, csv_fn=None, city_name="cd"):
        self.name = city_name.upper() + self.name
        self.model_dirname = timeDirName(self.model_dirname, is_mk=True)
        self.slog.__init__(os.path.join(self.model_dirname, "{0}_log.txt".format(self.name)), mode="a", )
        self.slog.kw("NAME", self.name)
        self.slog.kw("MODEL_DIRNAME", self.model_dirname)
        if csv_fn is not None:
            csv_fn = self.slog.kw("CSV_FN", csv_fn)
        else:
            # csv_fn = self.slog.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shadow1samples\spl.csv")
            # csv_fn = self.slog.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\15\train_data_spl2.csv")
            # csv_fn = self.slog.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\16\bjRF\train_data.csv")
            csv_fn = self.slog.kw("CSV_FN", r"F:\ProjectSet\Shadow\Hierarchical\Samples\21\sh2_spl21_sh2spl1.csv")
        self.shh_mis.initVHLColorTable()
        self.slog.kw("CATEGORY_NAMES", self.shh_mis.category_names)
        self.slog.kw("COLOR_TABLE", self.shh_mis.color_table)
        self.shh_mis.initPandas(pd.read_csv(csv_fn))
        self.shh_mis.dfFilterEQ("CITY", city_name)
        self.slog.kw("shh_mis.df.keys()", list(self.shh_mis.df.keys()))

        map_dict = {11: 2, 21: 1, 31: 2, 41: 3, 12: 3, 22: 3, 32: 3, 42: 3}
        # map_dict = None
        self.slog.kw("MAP DICT:", map_dict)
        self.slog.kw("Category Field Name:", self.shh_mis.initCategoryField(map_dict=map_dict))
        fit_keys = [
            'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
            # 'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH',
            "NDVI", "NDWI", "MNDWI",
            'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent',
        ]
        self.fit_keys = fit_keys

        self.slog.kw("fit_keys", fit_keys)
        self.shh_mis.addNDVIFeatExt()
        self.shh_mis.addNDWIFeatExt()
        self.shh_mis.addMNDWIFeatExt()
        self.shh_mis.initXKeys(fit_keys)
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
            fns = self.slog.kw("GEO_FNS", SHHConfig.SHH2_QD1_FNS)
            self.shh_mis.imdc(to_imdc_fn, fns)
        elif city_name == "bj":
            tiles_dirname = self.slog.kw(
                "TILES_DIRNAME", r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1_grayglcm")
            self.shh_mis.imdcTiles(to_imdc_fn, tiles_dirname=tiles_dirname)
        elif city_name == "cd":
            tiles_dirname = self.slog.kw(
                "TILES_DIRNAME", r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1_grayglcm")
            self.shh_mis.imdcTiles(to_imdc_fn, tiles_dirname=tiles_dirname)

        return self.shh_mis.clf


class VHLMain:

    def __init__(self):
        self.smip = SHHModImPytorch()

        def func_predict(model, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_predict = func_predict
        self.city_name = "cd"

    def main(self, is_train=False, is_imdc=False):
        self.smip.model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\VHLModels"
        self.smip.model_name = "VHLModel_VIT1"
        self.smip.epochs = 100
        self.smip.device = "cuda:0"
        self.smip.n_test = 5
        self.smip.batch_size = 32
        self.smip.class_names = SHHConfig.SHH_CNAMES_VHL
        self.smip.n_class = len(self.smip.class_names)
        self.smip.win_size = (8, 8)
        self.smip.model = VHLModel_VIT1().to(self.smip.device)
        self.smip.func_predict = self.func_predict
        self.smip.func_y_deal = lambda y: y + 1
        self.smip.initVHLColorTable()

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
        self.smip.train_ds, self.smip.test_ds = loadDS(self.smip.win_size, mod_dirname=self.smip.model_dirname,
                                                       city_name=self.city_name)
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
        self.smip.train()

    def imdc(self):
        self.smip.loadPTH(None)
        # grc: GDALRasterChannel = GDALRasterChannel()
        # grc.addGDALDatas(SHHConfig.SHH2_QD1_FNS[0])
        # self.smip.imdc(grc=grc, is_jdt=True, data_deal=data_deal)
        # self.smip.imdcTiles(
        #     # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
        #     tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1_retile",
        #     data_deal=data_deal,
        # )
        self.smip.imdcTiles(
            # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
            tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1_retile",
            data_deal=data_deal,
        )
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


class VHLFuncs:

    def __init__(self):
        self.name = "name"

    def samplingCSV(self):
        shh_spl = SHH2Samples()
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\17\sh2_spl17_cdvhl1.csv"
        to_csv_fn = numberfilename(csv_fn, sep="_spl")
        print("csv_fn:", csv_fn)
        print("to_csv_fn:", to_csv_fn)
        shh_spl.addCSV(csv_fn)
        shh_spl.initXY()
        shh_spl.sampling(grs=SHHConfig.GRS_SHH2_IMAGE1_FNS(), is_to_field=True)
        shh_spl.sampling(grs=SHHConfig.GRS_SHH2_IMAGE1_GLCM_FNS(), is_to_field=True)
        shh_spl.toCSV(to_csv_fn)

    def glcmImageRange(self):
        json_dict = readJson(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\qd_sh2_1.range")
        images_dict = {
            "AS_VV": r'F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SAR_glcm\as_vv\qd_sh2_1_as_vv',
            "AS_VH": r'F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SAR_glcm\as_vh\qd_sh2_1_as_vh',
            "DE_VV": r'F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SAR_glcm\de_vv\qd_sh2_1_de_vv',
            "DE_VH": r'F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SAR_glcm\de_vh\qd_sh2_1_de_vh',
        }

        json_dict = readJson(r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\bj_sh2_1.range")
        images_dict = {
            "AS_VV": r"G:\ImageData\SHH2BeiJingImages\sar_glcm\bj_sh2_1_as_vv",
            "AS_VH": r"G:\ImageData\SHH2BeiJingImages\sar_glcm\bj_sh2_1_as_vh",
            "DE_VV": r"G:\ImageData\SHH2BeiJingImages\sar_glcm\bj_sh2_1_de_vv",
            "DE_VH": r"G:\ImageData\SHH2BeiJingImages\sar_glcm\bj_sh2_1_de_vh",
        }

        json_dict = readJson(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1.range")
        images_dict = {
            "AS_VV": r"G:\ImageData\SHH2ChengDuImages\sar_glcm\cd_sh2_1_as_vv",
            "AS_VH": r"G:\ImageData\SHH2ChengDuImages\sar_glcm\cd_sh2_1_as_vh",
            "DE_VV": r"G:\ImageData\SHH2ChengDuImages\sar_glcm\cd_sh2_1_de_vv",
            "DE_VH": r"G:\ImageData\SHH2ChengDuImages\sar_glcm\cd_sh2_1_de_vh",
        }

        for k in images_dict:
            fn = images_dict[k]
            gr = GDALRaster(fn)
            d = gr.readAsArray()
            d = d.clip(json_dict[k]["min"], json_dict[k]["max"])
            gr.save(d.astype("float32"), save_geo_raster_fn=fn, dtype=gdal.GDT_Float32, descriptions=[k])
            print(fn)

    def accuracyImageCSV(self):
        gacm = GDALAccuracyConfusionMatrix(3, SHHConfig.SHH_CNAMES_VHL)
        #     imdc_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240407H133608\VHLModel_VIT1_epoch98_imdc1.tif",
        #             # imdc_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H195636\BJ_VHLML_imdc.tif",
        # r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H195636\BJ_VHLML_train_data.csv",
        gacm.addImageCSV(
            # imdc_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H192733\VHLModel_VIT1_epoch90_imdc1.tif",
            imdc_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240405H205015\QD_VHLML_imdc.tif",
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240406H192733\train_data.csv",
            filter_eqs={"TEST": 0}
        )

        print(gacm.cm.fmtCM())


def main():
    def func1():
        concatCSV(
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\15\spl_spl1.csv",
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\15\train_data_spl1.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\15\train_data_spl2.csv",
        )

    def func2():
        spl_txt_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\16\shh2_spl16_bjvhl1_qjy1.txt"
        to_spl_txt_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\16\shh2_spl16_bjvhl1_qjy2.txt"
        df = spl_txt_fn_sort(spl_txt_fn)

        df_spl_txt_func(df, spl_txt_fn, to_spl_txt_fn)
        return

    def spl_txt_fn_sort(spl_txt_fn):
        to_dict = splTxt2Dict(spl_txt_fn)
        csv_tmp_fn = spl_txt_fn + "-tmp2.csv"
        savecsv(csv_tmp_fn, to_dict)
        df = pd.read_csv(csv_tmp_fn)
        df = df.sort_values(by=["IS_TAG", "CATEGORY_CODE", "ESA", "SUM_RGB"], ascending=[False, True, True, False])
        df = DataFrameDictSort(df).sort(
            by=["IS_TAG", "CATEGORY_CODE", "ESA", "NDVI"], ascending=[False, True, True, False],
            filter_func=lambda line: line["CATEGORY_CODE"] == 1
        ).toDF()
        df.to_csv(csv_tmp_fn, index=False)
        return df

    def df_spl_txt_func(df, spl_txt_fn, to_spl_txt_fn):
        init_lines = []
        with open(spl_txt_fn, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                init_lines.append(line + "\n")
                if line == "> DATA":
                    break
        with open(to_spl_txt_fn, "w", encoding="utf-8", newline="") as f:
            f.writelines(init_lines)
            f.write("\n")
            cw = csv.writer(f)
            df_dict = df.to_dict("records")
            for i, line in enumerate(df_dict):
                line_list = list(line.values())[6:]
                to_line = [i + 1, line["CATEGORY_NAME"], line["IS_TAG"], line["X"], line["Y"], *line_list]
                cw.writerow(to_line)

    def read_spl_txt(spl_txt_fn):
        to_dict = splTxt2Dict(spl_txt_fn)
        csv_tmp_fn = spl_txt_fn + "-tmp.csv"
        savecsv(csv_tmp_fn, to_dict)
        df = pd.read_csv(csv_tmp_fn)
        return df

    def read_spl_txt_category(spl_txt_fn, is_name_to_code=True):
        sif_rw = SRTInfoFileRW(spl_txt_fn)
        to_dict = sif_rw.readAsDict(["CATEGORY"])
        name_to_code = {}
        for category_str in to_dict["CATEGORY"]:
            category_str = category_str.strip()
            clist1 = category_str.split(":", maxsplit=1)
            c_code = int(clist1[0])
            clist2 = clist1[1].split("|", maxsplit=1)
            c_color = eval(clist2[0])
            name = clist2[1].strip()
            name_to_code[name] = c_code
        if is_name_to_code:
            return name_to_code
        else:
            code_to_name = {}
            for k, data in name_to_code.items():
                code_to_name[data] = k
            return code_to_name

    def func3():
        spl_txt_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\16\shh2_spl16_bjvhl1_qjy27.txt"
        sh1_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shadow1samples\spl.csv"
        df_spl_txt = read_spl_txt(spl_txt_fn)
        df_spl_txt = df_spl_txt[df_spl_txt["IS_TAG"]]
        df_spl_txt["TEST"] = 1
        csv_tmp_fn = spl_txt_fn + "-tmp.csv"
        df_spl_txt.to_csv(csv_tmp_fn)
        df_shh1 = pd.read_csv(sh1_csv_fn)
        df_shh1 = df_shh1[df_shh1["CITY"] == "bj"]
        df_shh1["CATEGORY_CODE"] = categoryMap(df_shh1["CATEGORY"].tolist(), SHHConfig.CATE_MAP_VHL_82)
        csv_tmp_fn2 = spl_txt_fn + "-tmp2.csv"
        df_shh1.to_csv(csv_tmp_fn2, index=False)

        df = pd.DataFrame(concatCSV([csv_tmp_fn, csv_tmp_fn2]).data())
        df["CATEGORY"] = df["CATEGORY_CODE"]
        df.to_csv(csv_tmp_fn2, index=False)
        vhl_main = VHLMLMain()
        vhl_main.train(csv_tmp_fn2, city_name="bj")

        df_spl_txt = read_spl_txt(spl_txt_fn)
        df_spl_txt["NDWI"] = (df_spl_txt["B3"] - df_spl_txt["B8"]) / (df_spl_txt["B3"] + df_spl_txt["B8"])
        df_spl_txt["MNDWI"] = (df_spl_txt["B3"] - df_spl_txt["B12"]) / (df_spl_txt["B3"] + df_spl_txt["B12"])
        x = df_spl_txt[vhl_main.fit_keys].values
        y = vhl_main.shh_mis.clf.predict(x)
        df_spl_txt["CATEGORY_CODE"] = y
        code_to_name = read_spl_txt_category(spl_txt_fn, is_name_to_code=False)
        y_name = [code_to_name[y0] for y0 in y]
        df_spl_txt["CATEGORY_NAME"] = y_name
        df_spl_txt2 = read_spl_txt(spl_txt_fn)
        df_spl_txt.loc[df_spl_txt["IS_TAG"], "CATEGORY_NAME"] = df_spl_txt2[df_spl_txt2["IS_TAG"]]["CATEGORY_NAME"]
        df_spl_txt.loc[df_spl_txt["IS_TAG"], "CATEGORY_CODE"] = df_spl_txt2[df_spl_txt2["IS_TAG"]]["CATEGORY_CODE"]

        to_spl_txt_fn = numberfilename(spl_txt_fn)
        df_spl_txt_func(df_spl_txt, spl_txt_fn, to_spl_txt_fn)
        df_spl_txt.to_csv(csv_tmp_fn2)
        df = spl_txt_fn_sort(to_spl_txt_fn)
        df_spl_txt_func(df, to_spl_txt_fn, to_spl_txt_fn)
        print(to_spl_txt_fn)

        return

    def func4():
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\17\sh2_spl17_cdvhl1_spl1.csv"
        df = pd.read_csv(csv_fn)

        def norm(name1, name2):
            return (df[name1] - df[name2]) / (df[name1] + df[name2])

        df["NDVI"] = norm("B8", "B4")
        df["NDWI"] = norm("B3", "B8")
        df["MNDWI"] = norm("B3", "B12")

        csv_fn_tmp = csv_fn + "-tmp.csv"
        df.to_csv(csv_fn_tmp, index=False)

    def func5():
        samplingSHH21OptSarGLCM(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\Release\shadow1samples\spl.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\21\sh2_spl21_sh2spl1.csv",
        )

    # func5()

    VHLMain().main(is_train=True)
    # VHLFuncs().samplingCSV()
    # VHLFuncs().accuracyImageCSV()
    # VHLMLMain().train(city_name="cd")
    # loadDS((21,21), mod_dirname=r"F:\ProjectSet\Shadow\Hierarchical\VHLModels\20240403H105412")

    return


if __name__ == "__main__":
    """
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHVHL import VHLMain; VHLMain().main(is_imdc=True)" 
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHVHL import VHLMain; VHLMain().main(is_train=True)"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHVHL import VHLMLMain; VHLMLMain().train()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHVHL import VHLFuncs; VHLFuncs().glcmImageRange()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHVHL import VHLFuncs; VHLFuncs().samplingCSV()"

    """
    main()
