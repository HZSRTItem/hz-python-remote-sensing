# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2DL.py
@Time    : 2024/6/6 10:26
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2DL
-----------------------------------------------------------------------------"""
import os
import sys
import time
import warnings
from datetime import datetime
from shutil import copyfile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALUtils import GDALNumpySampling
from SRTCodes.NumpyUtils import NumpyDataCenter, TensorSelectNames
from SRTCodes.SRTModelImage import SRTModImPytorch
from SRTCodes.Utils import SRTLog, DirFileName, FN, Jdt, filterFileEndWith, writeTexts, saveJson, SRTWriteText, \
    changext, readJson, RunList
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Config import samplesDescription
from Shadow.Hierarchical.SHH2DLModels import Model, SHH2MOD_SpectralTextureDouble
from Shadow.Hierarchical.SHH2Draw import SHH2DrawTR
from Shadow.Hierarchical.SHH2Sample import samplingCSVData

LOG = SRTLog()
CITY_NAME = "qd"


class DLDataset(Dataset):

    def __init__(self, win_size, read_size):
        self.data_list = []
        self.y_list = []
        if win_size == read_size:
            self.ndc = None
        else:
            self.ndc = NumpyDataCenter(3, win_size, read_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        x = self.data_list[item]
        if self.ndc is not None:
            x = self.ndc.fit(x)
        y = self.y_list[item]
        x, y = data_deal(x, y)
        return x, y


def data_deal(x, y=None):
    if y is None:
        return x
    return x, y


def sampling(csv_fn, win_rows, win_columns):

    to_fn, to_npy_fn = getFileName(csv_fn, win_columns, win_rows)

    samplingCSVData(csv_fn, to_fn, to_npy_fn, win_rows, win_columns)


def getFileName(csv_fn, win_columns, win_rows):
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL")
    to_fn = dfn.fn("{}-{}_{}.csv".format(FN(csv_fn).getfilenamewithoutext(), win_rows, win_columns))
    to_npy_fn = FN(to_fn).changext("-data.npy")
    return to_fn, to_npy_fn


def csvFN(city_name):
    if city_name == "qd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl.csv"
    elif city_name == "bj":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv"
    elif city_name == "cd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl2.csv"
    else:
        csv_fn = None
    return csv_fn


def loadDS(win_size, to_csv_fn=None, city_name=None):
    if city_name is None:
        city_name = LOG.kw("CITY_NAME", CITY_NAME)
    else:
        city_name = LOG.kw("CITY_NAME", city_name)

    csv_fn = LOG.kw("CSV_FN", csvFN(city_name))
    read_size = 21, 21
    csv_fn_spl, npy_fn = getFileName(csv_fn, *read_size)

    if not os.path.isfile(csv_fn_spl):
        sampling(csv_fn, *read_size)
    df = pd.read_csv(csv_fn_spl)

    print(samplesDescription(df))

    if to_csv_fn is not None:
        df.to_csv(csv_fn_spl, index=False)
    data = np.load(npy_fn).astype("float32")

    train_ds = DLDataset(win_size, read_size)
    test_ds = DLDataset(win_size, read_size)

    map_dict_4 = {
        "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3
    }
    vhl_map_dict = {
        "IS": 0, "VEG": 1, "SOIL": 0, "WAT": 3,
        "IS_SH": 2, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 2
    }
    is_map_dict = {"IS": 1, "SOIL": 0, }
    ws_map_dict = {
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3
    }

    map_dict = vhl_map_dict

    def map_category(_category):
        if _category in map_dict:
            return map_dict[_category]
        return None

    def filter_data(n):
        if df["city"][n] != city_name:
            return None
        _category = map_category(str(df["CNAME"][n]))
        if _category is None:
            return None
        return _category

    for i in range(len(df)):
        if df["TEST"][i] == 1:
            category = filter_data(i)
            if category is not None:
                train_ds.data_list.append(data[i])
                train_ds.y_list.append(category)
        else:
            category = filter_data(i)
            if category is not None:
                test_ds.data_list.append(data[i])
                test_ds.y_list.append(category)

    return train_ds, test_ds


def buildModel(build_model_dict):
    def func1():
        tsn = TensorSelectNames(*SHH2Config.NAMES, dim=1)
        tsn.addTSN("OPT", ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", ])
        tsn.addTSN("AS_BS", ["AS_VV", "AS_VH"])
        tsn.addTSN("DE_BS", ["DE_VV", "DE_VH"])
        tsn.addTSN("AS_C2", ["AS_C11", "AS_C22"])
        tsn.addTSN("DE_C2", ["DE_C11", "DE_C22"])
        tsn.addTSN("AS_HA", ["AS_H", "AS_Alpha"])
        tsn.addTSN("DE_HA", ["DE_H", "DE_Alpha"])
        model = Model(tsn=tsn)

        def sar_deal(name, x):
            return (model.tsn[name].fit(x) + 20) / 30

        def func(x):
            x_opt = model.tsn["OPT"].fit(x)
            x_opt = x_opt / 3000

            x_as_bs = sar_deal("AS_BS", x)
            x_de_bs = sar_deal("DE_BS", x)
            x_as_c2 = sar_deal("AS_C2", x)
            x_de_c2 = sar_deal("DE_C2", x)

            x_as_ha = model.tsn["AS_HA"].fit(x)
            x_as_ha[:, 1] = x_as_ha[:, 1] / 90
            x_de_ha = model.tsn["DE_HA"].fit(x)
            x_de_ha[:, 1] = x_de_ha[:, 1] / 90

            x = torch.cat([x_opt, x_as_bs, x_de_bs, x_as_c2, x_de_c2, x_as_ha, x_de_ha], dim=1)
            return x

        model.xforward = func

        return model

    def func2():
        tsn = TensorSelectNames(*SHH2Config.NAMES, dim=1)
        tsn.addTSN("OPT", ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", ])
        tsn.addTSN("BS", ["AS_VV", "DE_VV", "AS_VH", "DE_VH"])
        tsn.addTSN("C2", ["AS_C11", "DE_C11", "AS_C22", "DE_C22"])
        tsn.addTSN("HA", ["AS_H", "DE_H", "AS_Alpha", "DE_Alpha"])

        ndc3 = NumpyDataCenter(4, (3, 3), (21, 21))

        def to_3d(x):
            x1 = torch.zeros(x.shape[0], 2, 2, x.shape[2], x.shape[3]).to(x.device)
            x1[:, 0, :, :, :] = x[:, [0, 1], :, :]
            x1[:, 1, :, :, :] = x[:, [2, 3], :, :]
            return x1

        def xforward(x: torch.Tensor):
            x_opt = tsn["OPT"].fit(x)
            x_bs = tsn["BS"].fit(x)
            x_c2 = tsn["C2"].fit(x)
            x_ha = tsn["HA"].fit(x)
            x0 = ndc3.fit(torch.cat([x_opt, x_bs, x_c2, x_ha], dim=1))
            return x0, x_opt[:, [2, 3], :, :], to_3d(x_bs), to_3d(x_c2), to_3d(x_ha)

        model = SHH2MOD_SpectralTextureDouble(tsn.length(), 4, blocks_type="Transformer", is_texture=True)
        model.xforward = xforward

        return model

    def func3():
        tsn = TensorSelectNames(*SHH2Config.NAMES, dim=1)
        tsn.addTSN("OPT", ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", ])
        tsn.addTSN("BS", ["AS_VV", "DE_VV", "AS_VH", "DE_VH"])
        tsn.addTSN("C2", ["AS_C11", "DE_C11", "AS_C22", "DE_C22"])
        tsn.addTSN("HA", ["AS_H", "DE_H", "AS_Alpha", "DE_Alpha"])

        ndc3 = NumpyDataCenter(4, (3, 3), (21, 21))

        def to_3d(x):
            x1 = torch.zeros(x.shape[0], 2, 2, x.shape[2], x.shape[3]).to(x.device)
            x1[:, 0, :, :, :] = x[:, [0, 1], :, :]
            x1[:, 1, :, :, :] = x[:, [2, 3], :, :]
            return x1

        def xforward(x: torch.Tensor):
            x_opt = tsn["OPT"].fit(x)
            x_bs = tsn["BS"].fit(x)
            x_c2 = tsn["C2"].fit(x)
            x_ha = tsn["HA"].fit(x)
            x0 = ndc3.fit(torch.cat([x_opt, x_bs, x_c2, x_ha], dim=1))
            return x0, x_opt[:, [2, 3], :, :], to_3d(x_bs), to_3d(x_c2), to_3d(x_ha)

        model = SHH2MOD_SpectralTextureDouble(
            in_channels=tsn.length(),
            num_category=4,
            blocks_type=build_model_dict["model"]["spectral"],
            is_texture=build_model_dict["model"]["texture"],
        )

        model.xforward = xforward

        return model

    return func2()


class DeepLearning:
    """ DeepLearning

    change:
        self.name: save name
        self.smip.model_dirname: model save dir name
        self.smip.class_names: class names
        self.smip.win_size: win size of data predict
        self.color_table: color table of image classification

    """

    def __init__(self, color_table=None, city_name=None):
        if color_table is None:
            color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
            color_table = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255), 4: (0, 0, 0), }
            color_table = {1: (0, 0, 0), 2: (255, 0, 0)}
            color_table = {1: (128, 0, 0), 2: (0, 128, 0), 3: (128, 128, 0), 4: (0, 0, 128), }
            color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 0), 4: (0, 0, 255), }
            color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 0), 4: (0, 0, 255), }
            # color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }

        self.name = "SHH2DL"
        self.smip = SRTModImPytorch()

        if city_name is None:
            city_name = CITY_NAME
        self.city_name = city_name

        def func_predict(model, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_predict = func_predict
        self.color_table = color_table

        # writeTexts(os.path.join(self.smip.model_dirname, "sys.argv.txt"), sys.argv, mode="a")
        self.slog = SRTLog()

    def main(
            self,
            model_name="Model",
            build_model_dict=None,
            class_names=("NOT_KNOW", "IS", "VEG", "SOIL", "WAT"),
            win_size=(21, 21),
            model_dirname=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods",
            epochs=100,
    ):
        self.smip.model_name = model_name
        self.smip.class_names = list(class_names)

        self.smip.win_size = win_size
        self.smip.model = buildModel(build_model_dict).to(self.smip.device)

        self.smip.model_dirname = model_dirname
        self.smip.epochs = epochs
        self.smip.device = "cuda:0"
        self.smip.n_test = 5
        self.smip.batch_size = 32
        self.smip.n_class = len(self.smip.class_names)
        self.smip.func_predict = self.func_predict
        self.smip.func_y_deal = lambda y: y
        self.smip.initColorTable(self.color_table)
        self.smip.n = 12000
        return

    def train(self):
        self.smip.timeDirName()
        LOG.__init__(os.path.join(self.smip.model_dirname, "SHH2DL_LOG.txt"))
        self.slog = SRTLog(os.path.join(self.smip.model_dirname, "log.txt"))
        self.smip.train_ds, self.smip.test_ds = loadDS(self.smip.win_size, self.smip.toCSVFN(), self.city_name)
        self.smip.initTrainLog()
        self.smip.initPytorchTraining()
        self.smip.pt.func_logit_category = self.func_predict
        self.smip.pt.func_y_deal = lambda y: y + 1
        self.smip.initModel()
        self.slog.kw("self.smip.model.tsn.names", self.smip.model.tsn.toDict())
        self.smip.initDataLoader()
        self.smip.initCriterion(nn.CrossEntropyLoss())
        self.smip.initOptimizer(torch.optim.Adam, lr=0.001, eps=10e-5)
        self.smip.pt.addScheduler(torch.optim.lr_scheduler.StepLR(self.smip.pt.optimizer, 20, gamma=0.5, last_epoch=-1))
        self.smip.copyFile(__file__)
        self.smip.copyFile(r"F:\PyCodes\Shadow\Hierarchical\SHH2DLModels.py")
        self.smip.print()
        self.smip.train()
        return self.smip.model_dirname

    def imdc(self, mod_fn=None):
        if mod_fn is None:
            mod_fn = sys.argv[2]
        self.smip.loadPTH(mod_fn)

        # grc: GDALRasterChannel = GDALRasterChannel()
        # grc.addGDALDatas("")
        # self.smip.imdc(grc=grc, is_jdt=True, data_deal=data_deal)

        def tiles():
            _to_fn = None
            if self.city_name == "qd":
                _to_fn = self.smip.imdcTiles(
                    tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles",
                    data_deal=data_deal,
                )
            elif self.city_name == "bj":
                _to_fn = self.smip.imdcTiles(
                    tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles",
                    data_deal=data_deal,
                )
            elif self.city_name == "cd":
                _to_fn = self.smip.imdcTiles(
                    tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles",
                    data_deal=data_deal,
                )
            return _to_fn

        def raster():
            _to_fn = None
            if self.city_name == "qd":
                _to_fn = self.smip.imdcGDALFile(fn=SHH2Config.QD_ENVI_FN, data_deal=data_deal, is_print=True, )
            elif self.city_name == "bj":
                _to_fn = self.smip.imdcGDALFile(fn=SHH2Config.BJ_ENVI_FN, data_deal=data_deal, is_print=True, )
            elif self.city_name == "cd":
                _to_fn = self.smip.imdcGDALFile(fn=SHH2Config.CD_ENVI_FN, data_deal=data_deal, is_print=True, )
            return _to_fn

        # self.smip.imdcTiles(
        #     # to_fn=r"F:\Week\20240331\Data\20240329H185618\Net2_epoch2_imdc5.tif",
        #     tiles_dirname=r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\cd_sh2_1_retile",
        #     data_deal=data_deal,
        # )

        to_fn = raster()
        return to_fn

    def imdc2(self, mod_fn=None):
        if mod_fn is None:
            mod_fn = sys.argv[2]
        self.smip.loadPTH(mod_fn)
        to_dirname = os.path.splitext(mod_fn)[0] + "_TRImdc"
        if not os.path.isdir(to_dirname):
            os.mkdir(to_dirname)

        if self.city_name == "qd":
            fns = filterFileEndWith(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions", ".tif")
        elif self.city_name == "bj":
            fns = filterFileEndWith(r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\TestRegions", ".tif")
        elif self.city_name == "cd":
            fns = filterFileEndWith(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\TestRegions", ".tif")
        else:
            fns = []

        self.smip.imdcGDALFiles(to_dirname=to_dirname, fns=fns, data_deal=data_deal, )
        return to_dirname

    def imdc3(self, mod_fn=None):
        if mod_fn is None:
            mod_fn = sys.argv[2]
        self.smip.loadPTH(mod_fn)
        print("mod_fn", mod_fn)
        self.smip.imdcGDALFile(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles\SHH2_QD2_envi_3_4.tif",
                               is_print=True)


def DeepLearning_main(is_train=False):
    dl = DeepLearning()
    dl.main()
    if is_train:
        dl.train()
    else:
        to_dirname = dl.imdc2(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240628H164435\Model_epoch86.pth")
        SHH2DrawTR(CITY_NAME).addImdcDirName("Imdc", dirname=to_dirname).draw18("NRG", "Imdc").show()


def show():
    SHH2DrawTR("qd").addImdcDirName(
        "Imdc", dirname=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\qd\STD_epoch78_TRImdc"
    ).draw18("NRG", "Imdc").show()

    # SHH2DrawTR("qd").addImdc(
    #     "Imdc", r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\qd\STD_epoch30_imdc1.tif"
    # ).draw18("NRG", "Imdc").show()


def main():
    DeepLearning_main()
    return


def run(is_run=False):
    json_fn = r"F:\ProjectSet\Shadow\Hierarchical\Run\VHL_ST.json"

    if not is_run:
        run_dict = [
            {"run": False, "type": "training", "city_name": "qd", "model": {"spectral": "CNN", "texture": False}},
            {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 1}},
            {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 5}},
            {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 10}},
            {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 30}},
            {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 60}},
            {"run": False, "type": "imdc", "city_name": "qd", "imdc": {"models": 90}},
            {"run": False, "type": "training", "city_name": "qd",
             "model": {"spectral": "Transformer", "texture": False}},
            {"run": False, "type": "training", "city_name": "qd", "model": {"spectral": "CNN", "texture": True}},
            {"run": False, "type": "training", "city_name": "qd",
             "model": {"spectral": "Transformer", "texture": True}},
        ]
        run_dict = []
        for city_name in ["qd", "bj", "cd"]:
            for train_type in [
                {"spectral": "CNN", "texture": False},
                {"spectral": "Transformer", "texture": False},
                {"spectral": "CNN", "texture": True},
                {"spectral": "Transformer", "texture": True},
            ]:
                run_dict.append({"run": False, "type": "training", "city_name": city_name, "model": train_type})
                for imdc_mod in [2, 10, 90]:
                    run_dict.append(
                        {"run": False, "type": "imdc", "city_name": city_name, "imdc": {"models": imdc_mod}})

        print(run_dict)
        runlist_fn = r"F:\ProjectSet\Shadow\Hierarchical\Run\VHL_ST_runlist.txt"
        writeTexts(runlist_fn, "", mode="w")
        for i in run_dict:
            writeTexts(runlist_fn,
                       "python -c \"import sys; sys.path.append(r'F:\\PyCodes'); "
                       "from Shadow.Hierarchical.SHH2DL import run; run(True)\" %* \n",
                       mode="a")
        writeTexts(runlist_fn, "\n", mode="a")
        saveJson(run_dict, json_fn)

    else:

        model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods"

        dfn = DirFileName(os.path.split(json_fn)[0])
        sw = SRTWriteText(dfn.fn("VHL_ST_save.txt"), mode="a")
        to_json_fn = changext(json_fn, "_run.json")

        if not os.path.isfile(to_json_fn):
            copyfile(json_fn, to_json_fn)

        json_dict = readJson(to_json_fn)

        n_run = -1
        run_dict = {}
        for i in range(len(json_dict)):
            if not json_dict[i]["run"]:
                n_run = i
                run_dict = json_dict[i]
                break

        if n_run == -1:
            print("#", "-" * 10, "End Run", n_run, "-" * 10, "#")
            return

        print("#", "-" * 10, "Run", n_run, "->", len(json_dict), "-" * 10, "#")

        if run_dict["type"] == "training":
            time.sleep(1)
            current_time = datetime.now()
            city_name = None
            if run_dict["city_name"] == "qd":
                city_name = "QingDao"
            elif run_dict["city_name"] == "bj":
                city_name = "BeiJing"
            elif run_dict["city_name"] == "cd":
                city_name = "ChengDu"

            dl = DeepLearning(city_name=run_dict["city_name"])
            dl.main(
                model_name="VHL_ST",
                build_model_dict=run_dict,
                class_names=("NOT_KNOW", "IS", "VEG", "SOIL", "WAT"),
                win_size=(21, 21),
                model_dirname=model_dirname,
                epochs=100,
            )
            to_dirname = dl.train()

            sw.write("{}\n{} DL VHL4 {} {}\n".format(
                current_time.strftime("%Y年%m月%d日%H:%M:%S"), city_name,
                run_dict["model"], to_dirname
            ))

            saveJson(run_dict, dfn.fn("build_model_dict.json"))
            writeTexts(dfn.fn("to_dirname.txt"), to_dirname)
            print("training")

        elif run_dict["type"] == "imdc":

            with open(dfn.fn("to_dirname.txt"), "r", encoding="utf-8") as f:
                to_dirname = f.read()

            build_model_dict = readJson(dfn.fn("build_model_dict.json"))
            model_name = "VHL_ST"

            dl = DeepLearning(city_name=run_dict["city_name"])
            dl.main(
                model_name=model_name,
                build_model_dict=build_model_dict,
                class_names=("NOT_KNOW", "IS", "VEG", "SOIL", "WAT"),
                win_size=(21, 21),
                model_dirname=model_dirname,
                epochs=100,
            )
            mod_fn = os.path.join(to_dirname, "{}_epoch{}.pth".format(model_name, run_dict["imdc"]["models"]))
            print("Imdc: ", mod_fn)
            dl.imdc(mod_fn)

        json_dict[n_run]["run"] = True
        saveJson(json_dict, to_json_fn)


def imdc(is_run=False):
    rl = RunList()
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods")

    rl.add({"city_name": "bj", "model": {'spectral': 'CNN', 'texture': False}, "dirname": dfn.fn("20240630H105741"),
            "imdc": 89})
    rl.add({"city_name": "bj", "model": {'spectral': 'Transformer', 'texture': False},
            "dirname": dfn.fn("20240630H110340"), "imdc": 89})
    rl.add({"city_name": "bj", "model": {'spectral': 'CNN', 'texture': True}, "dirname": dfn.fn("20240630H111158"),
            "imdc": 89})
    rl.add(
        {"city_name": "bj", "model": {'spectral': 'Transformer', 'texture': True}, "dirname": dfn.fn("20240630H114524"),
         "imdc": 89})

    rl.add({"city_name": "cd", "model": {'spectral': 'CNN', 'texture': False}, "dirname": dfn.fn("20240630H122133"),
            "imdc": 89})
    rl.add({"city_name": "cd", "model": {'spectral': 'Transformer', 'texture': False},
            "dirname": dfn.fn("20240630H122728"), "imdc": 89})
    rl.add({"city_name": "cd", "model": {'spectral': 'CNN', 'texture': True}, "dirname": dfn.fn("20240630H123521"),
            "imdc": 89})
    rl.add(
        {"city_name": "cd", "model": {'spectral': 'Transformer', 'texture': True}, "dirname": dfn.fn("20240630H130543"),
         "imdc": 89})

    rl.add({"city_name": "qd", "model": {'spectral': 'CNN', 'texture': False}, "dirname": dfn.fn("20240630H133802"),
            "imdc": 89})
    rl.add({"city_name": "qd", "model": {'spectral': 'Transformer', 'texture': False},
            "dirname": dfn.fn("20240630H134415"), "imdc": 89})
    rl.add({"city_name": "qd", "model": {'spectral': 'CNN', 'texture': True}, "dirname": dfn.fn("20240630H135241"),
            "imdc": 89})
    rl.add(
        {"city_name": "qd", "model": {'spectral': 'Transformer', 'texture': True}, "dirname": dfn.fn("20240630H142608"),
         "imdc": 89})

    # rl.show()

    model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods"
    model_name = "VHL_ST"

    def run_func(_rl: RunList):
        run_dict = _rl.run_dict

        dl = DeepLearning(city_name=run_dict["city_name"])
        dl.main(
            model_name=model_name,
            build_model_dict=run_dict,
            class_names=("NOT_KNOW", "IS", "VEG", "SOIL", "WAT"),
            win_size=(21, 21),
            model_dirname=model_dirname,
            epochs=100,
        )
        mod_fn = os.path.join(run_dict["dirname"], "{}_epoch{}.pth".format(model_name, run_dict["imdc"]))
        print("Model   :", run_dict["model"])
        print("Imdc    :", mod_fn)
        print("Is Exist:", os.path.exists(mod_fn))
        _rl.sw.write("mod_fn:", mod_fn)
        to_fn = dl.imdc(mod_fn)
        _rl.sw.write("imdc:", to_fn)

    rl.fit(
        name="{}_imdc".format(model_name),
        dirname=r"F:\ProjectSet\Shadow\Hierarchical\Run",
        cmd_line=r'''python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL import imdc; imdc(True)" %* ''',
        is_run=is_run,
        run_func=run_func,
    )


def imdc2():
    dl = DeepLearning()
    dl.main(build_model_dict={"model": {'spectral': 'Transformer', 'texture': True}, })
    dl.imdc3(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240630H142608\VHL_ST_epoch61.pth")


def trainimdc(is_run=False):
    rl = RunList()
    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods")

    def rl_add_CNN_Transformer_texture():
        for city_name in ["qd", "bj", "cd"]:
            for model in [
                {"spectral": "CNN", "texture": False},
                {"spectral": "Transformer", "texture": False},
                {"spectral": "CNN", "texture": True},
                {"spectral": "Transformer", "texture": True},
            ]:
                rl.add({"type": "training", "city_name": city_name, "model": model})
                for imdc_mod in [1, 90]:
                    rl.add({"type": "imdc", "city_name": city_name, "model": model, "imdc": {"models": imdc_mod}})

    rl_add_CNN_Transformer_texture()
    if not is_run:
        rl.show()
    # sys.exit()

    model_dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods"
    model_name = "VHL4_ST"

    def run_func(_rl: RunList):
        run_dict = _rl.run_dict
        _rl.show(_rl.n_run)
        time.sleep(1)
        if run_dict["type"] == "training":
            dl = DeepLearning(city_name=run_dict["city_name"])
            dl.main(
                model_name=model_name,
                build_model_dict=run_dict,
                class_names=("NOT_KNOW", "IS", "VEG", "SH", "WAT"),
                win_size=(21, 21),
                model_dirname=model_dirname,
                epochs=100,
            )
            to_dirname = None
            to_dirname = dl.train()

            city_name = None
            if run_dict["city_name"] == "qd":
                city_name = "QingDao"
            elif run_dict["city_name"] == "bj":
                city_name = "BeiJing"
            elif run_dict["city_name"] == "cd":
                city_name = "ChengDu"
            current_time = datetime.now()
            _rl.sw.write("{}\n{} DL VHL4 {} {}\n".format(
                current_time.strftime("%Y年%m月%d日%H:%M:%S"), city_name,
                run_dict["model"], to_dirname
            ))

            _rl.saveJsonData("build_model_dict", run_dict)
            _rl.saveJsonData("to_dirname", to_dirname)

        elif run_dict["type"] == "imdc":

            build_model_dict = _rl.readJsonData("build_model_dict")
            to_dirname = _rl.readJsonData("to_dirname")

            dl = DeepLearning(city_name=run_dict["city_name"])
            dl.main(
                model_name=model_name,
                build_model_dict=build_model_dict,
                class_names=("NOT_KNOW", "IS", "VEG", "SH", "WAT"),
                win_size=(21, 21),
                model_dirname=model_dirname,
                epochs=100,
            )
            mod_fn = os.path.join(to_dirname, "{}_epoch{}.pth".format(model_name, run_dict["imdc"]["models"]))
            print("Model   :", run_dict["model"])
            print("Imdc    :", mod_fn)
            print("Is Exist:", os.path.exists(mod_fn))
            _rl.sw.write("mod_fn:", mod_fn)
            to_fn = None
            to_fn = dl.imdc(mod_fn)
            _rl.sw.write("imdc:", to_fn)
            _rl.sw.write()

    rl.fit(
        name="{}_trainimdc".format(model_name),
        dirname=r"F:\ProjectSet\Shadow\Hierarchical\Run",
        cmd_line=r'''python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL import trainimdc; trainimdc(True)" %* ''',
        is_run=is_run,
        run_func=run_func,
    )


if __name__ == "__main__":
    trainimdc()
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL import DeepLearning_main; DeepLearning_main(True)" 
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL import DeepLearning_main; DeepLearning_main(False)"
    """
