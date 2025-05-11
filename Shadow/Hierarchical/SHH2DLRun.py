# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2DLRun.py
@Time    : 2024/7/21 11:08
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2DLRun
-----------------------------------------------------------------------------"""
import os

import numpy as np
import torch
from torch import nn

from DeepLearning.DenseNet import densenet121
from DeepLearning.EfficientNet import efficientnet_b7, efficientnet_v2_m
from DeepLearning.GoogLeNet import GoogLeNet
from DeepLearning.InceptionV3 import Inception3
from DeepLearning.MobileNetV2 import MobileNetV2
from DeepLearning.MobileNetV3 import mobilenet_v3_small
from DeepLearning.ResNet import BasicBlock, ResNet
from DeepLearning.ResNet import resnext50_32x4d
from DeepLearning.ShuffleNetV2 import shufflenet_v2_x0_5
from DeepLearning.SqueezeNet import SqueezeNet
from DeepLearning.VisionTransformer import VisionTransformerChannel
from SRTCodes.GDALRasterIO import GDALRaster, saveGTIFFImdc
from SRTCodes.SRTLinux import W2LF
from SRTCodes.SRTModel import SamplesData, TorchModel
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import DirFileName, SRTWriteText, saveJson
from Shadow.Hierarchical import SHH2Config

_DL_SAMPLE_DIRNAME = W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL")
_DL_DFN = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods")

_X_KEYS = [
    #  0  1  2  3  4  5
    "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
    #  6  7  8  9 10 11
    "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
    # 12 13 14 15 16 17
    "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
]
_MAP_DICT_4 = {
    "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
    "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
}
_CM_NAME_4 = ["IS", "VEG", "SOIL", "WAT"]
_COLOR_TABLE_4 = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
_BACK_DIRNAME = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp"


def _GET_DL_MODEL(_mod_name, win_size, _in_ch, _n_category):
    if _mod_name == "ResNet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], in_ch=_in_ch, num_classes=4)

    if _mod_name == "VIT":
        return VisionTransformerChannel(
            in_channels=_in_ch, image_size=win_size[0], patch_size=2, num_layers=12,
            num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=_n_category,
        )

    if _mod_name == "SqueezeNet":
        return SqueezeNet(version="1_1", num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "GoogLeNet":
        return GoogLeNet(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "DenseNet121":
        return densenet121(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "Inception3":
        return Inception3(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "ShuffleNetV2X05":
        return shufflenet_v2_x0_5(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "MobileNetV2":
        return MobileNetV2(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "MobileNetV3Small":
        return mobilenet_v3_small(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "ResNeXt5032x4d":
        return resnext50_32x4d(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "EfficientNetB7":
        return efficientnet_b7(num_classes=_n_category, in_channels=_in_ch)

    if _mod_name == "EfficientNetV2M":
        return efficientnet_v2_m(num_classes=_n_category, in_channels=_in_ch)

    return None


def X_DEAL_18(_x):
    for i in range(0, 6):
        _x[i] = _x[i] / 1600
    # for i in range(6, 10):
    #     _x[i] = (_x[i] + 30) / 35
    # for i in range(12, 16):
    #     _x[i] = (_x[i] + 30) / 35
    # _x[11] = _x[11] / 90
    # _x[17] = _x[11] / 90
    return _x


def getCSVFn(city_name):
    if city_name == "qd":
        csv_fn = W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl2.csv")
        csv_fn = W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd6.csv")
    elif city_name == "cd":
        csv_fn = W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv")
    elif city_name == "bj":
        csv_fn = W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv")
    else:
        raise Exception("City name \"{}\"".format(city_name))
    return csv_fn


def run1(city_name):
    from torch import nn

    from SRTCodes.SRTModel import SamplesData, TorchModel
    from Shadow.Hierarchical import SHH2Config
    from DeepLearning.DenseNet import densenet121
    from DeepLearning.GoogLeNet import GoogLeNet
    from DeepLearning.InceptionV3 import Inception3
    from DeepLearning.ResNet import BasicBlock, ResNet
    from DeepLearning.SqueezeNet import SqueezeNet
    from DeepLearning.VisionTransformer import VisionTransformerChannel
    from DeepLearning.EfficientNet import efficientnet_b7, efficientnet_v2_m
    from DeepLearning.MobileNetV2 import MobileNetV2
    from DeepLearning.MobileNetV3 import mobilenet_v3_small
    from DeepLearning.ResNet import resnext50_32x4d
    from DeepLearning.ShuffleNetV2 import shufflenet_v2_x0_5

    from SRTCodes.SRTTimeDirectory import TimeDirectory
    from SRTCodes.Utils import DirFileName, RumTime

    _DL_SAMPLE_DIRNAME = W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL")

    def x_deal(_x):
        for i in range(0, 6):
            _x[i] = _x[i] / 1600
        for i in range(6, 10):
            _x[i] = (_x[i] + 30) / 35
        for i in range(12, 16):
            _x[i] = (_x[i] + 30) / 35
        _x[11] = _x[11] / 90
        _x[17] = _x[11] / 90
        return _x

    def get_model(_mod_name, _in_ch=None, _n_category=4):
        if _in_ch is None:
            _in_ch = len(get_names)

        if _mod_name == "ResNet18":
            return ResNet(BasicBlock, [2, 2, 2, 2], in_ch=len(get_names), num_classes=4)

        if _mod_name == "VIT":
            return VisionTransformerChannel(
                in_channels=_in_ch, image_size=win_size[0], patch_size=2, num_layers=12,
                num_heads=12, hidden_dim=768, mlp_dim=3072, num_classes=4,

            )

        if _mod_name == "SqueezeNet":
            return SqueezeNet(version="1_1", num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "GoogLeNet":
            return GoogLeNet(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "DenseNet121":
            return densenet121(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "Inception3":
            return Inception3(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "ShuffleNetV2X05":
            return shufflenet_v2_x0_5(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "MobileNetV2":
            return MobileNetV2(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "MobileNetV3Small":
            return mobilenet_v3_small(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "ResNeXt5032x4d":
            return resnext50_32x4d(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "EfficientNetB7":
            return efficientnet_b7(num_classes=_n_category, in_channels=_in_ch)

        if _mod_name == "EfficientNetV2M":
            return efficientnet_v2_m(num_classes=_n_category, in_channels=_in_ch)

        return None

    dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods")
    td = TimeDirectory(dfn.fn())
    td.initLog()
    td.kw("DIRNAME", td.time_dirname())

    init_model_name = "DL"
    td.log("\n#", "-" * 50, city_name.upper(), init_model_name, "-" * 50, "#\n")
    csv_fn = getCSVFn(city_name)
    raster_fn = SHH2Config.GET_RASTER_FN(city_name)
    get_names = [
        #  0  1  2  3  4  5
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
        #  6  7  8  9 10 11
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        # 12 13 14 15 16 17
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ]
    map_dict = {
        "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
    }
    cm_names = ["IS", "VEG", "SOIL", "WAT"]
    epochs = 100

    td.kw("CITY_NAME", city_name)
    td.kw("INIT_MODEL_NAME", init_model_name)
    td.kw("CSV_FN", csv_fn)
    td.kw("GET_NAMES", get_names)
    td.copyfile(csv_fn)
    td.copyfile(__file__)

    read_size = (25, 25)

    sd = SamplesData(_dl_sample_dirname=_DL_SAMPLE_DIRNAME)
    sd.addDLCSV(
        csv_fn, read_size, get_names, x_deal,
        grs={"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }
    )

    win_size_list = [
        ("ResNet18", (7, 7)),
        ("VIT", (24, 24)),
        ("SqueezeNet", (24, 24)),
        ("GoogLeNet", (24, 24)),
        # ("DenseNet121", (24, 24)),  # size small
        # ("Inception3", (24, 24)),  # size small
        ("ShuffleNetV2X05", (7, 7)),
        ("MobileNetV2", (7, 7)),
        ("MobileNetV3Small", (7, 7)),
        ("ResNeXt5032x4d", (7, 7)),
        ("EfficientNetB7", (7, 7)),
        ("EfficientNetV2M", (7, 7)),
    ]
    run_time = RumTime(len(win_size_list)).strat()

    for i_win_size, (init_model_name, win_size) in enumerate(win_size_list):
        model = get_model(init_model_name)
        model_name = "{}_{}-{}".format(init_model_name, win_size[0], win_size[1])
        dfn_tmp = DirFileName(td.fn(model_name))
        dfn_tmp.mkdir()

        td.log("\n#", "-" * 30, i_win_size + 1, model_name, "-" * 30, "#\n")
        torch_mod = TorchModel()
        torch_mod.filename = dfn_tmp.fn(model_name + ".hm")
        torch_mod.map_dict = map_dict
        torch_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        torch_mod.model = model
        torch_mod.criterion = nn.CrossEntropyLoss()
        torch_mod.win_size = win_size
        torch_mod.read_size = read_size
        torch_mod.epochs = 100
        torch_mod.n_epoch_save = -1
        torch_mod.train_filters.extend([("city", "==", city_name)])
        torch_mod.test_filters.extend([("city", "==", city_name)])
        torch_mod.cm_names = cm_names

        td.kw("TORCH_MOD.FILENAME", torch_mod.filename)
        td.kw("TORCH_MOD.MAP_DICT", torch_mod.map_dict)
        td.kw("TORCH_MOD.COLOR_TABLE", torch_mod.color_table)
        td.kw("TORCH_MOD.MODEL", torch_mod.model.__class__)
        td.kw("TORCH_MOD.CRITERION", torch_mod.color_table)
        td.kw("TORCH_MOD.WIN_SIZE", torch_mod.win_size)
        td.kw("TORCH_MOD.READ_SIZE", torch_mod.read_size)
        td.kw("TORCH_MOD.EPOCHS", torch_mod.epochs)
        td.kw("TORCH_MOD.N_EPOCH_SAVE", torch_mod.n_epoch_save)
        td.kw("TORCH_MOD.CM_NAMES", torch_mod.cm_names)

        model_sw = td.buildWriteText(r"{}\{}.txt".format(model_name, model_name), "a")
        model_sw.write(torch_mod.model)

        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts(td.log)

        line_sw = td.buildWriteText(r"{}\{}_training-log.txt".format(model_name, model_name), "a")
        to_list = []

        def func_field_record_save(field_records):
            line = field_records.line
            to_list.append(line.copy())

            if int(line["Accuracy"]) != -1:
                for k in line:
                    line_sw.write("| {}:{} ".format(k, line[k]), end="")
                line_sw.write("|")
                if line["Batch"] == 0:
                    td.log("+ Epoch:", "{:<6d}".format(line["Epoch"]), end=" ", is_print=False)
                    td.log("Loss:", "{:<12.6f}".format(line["Loss"]), end=" ", is_print=False)
                    td.log("Accuracy:", "{:>6.3f}".format(line["Accuracy"]), end="\n", is_print=False)

        torch_mod.func_field_record_save = func_field_record_save
        torch_mod.save_model_fmt = dfn_tmp.fn(model_name + "_" + city_name + "_{}.pth")
        td.kw("TORCH_MOD.SAVE_MODEL_FMT", torch_mod.save_model_fmt)
        torch_mod.train()

        torch_mod.imdc(
            raster_fn, data_deal=x_deal,
            mod_fn=None, read_size=(500, -1), is_save_tiles=True, fun_print=td.log
        )

        td.saveJson(r"{}\{}_training-log.json".format(model_name, model_name), to_list)
        # except Exception as e:
        #     print(e)

        run_time.add().printInfo()

    return


def run2(city_name, init_model_name, win_size, is_train):
    td = TimeDirectory(dfn.fn(), time_dirname=W2LF(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240809H221443"))
    td.initLog(mode="a")
    td.kw("DIRNAME", td.time_dirname())

    td.log("\n#", "-" * 50, city_name.upper(), init_model_name, "-" * 50, "#\n")
    csv_fn = getCSVFn(city_name)

    raster_fn = SHH2Config.GET_RASTER_FN(city_name)

    get_names = [
        #  0  1  2  3  4  5
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
        #  6  7  8  9 10 11
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        # 12 13 14 15 16 17
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ]
    map_dict = {
        "IS": 0,
        "VEG": 1,
        "SOIL": 2,
        "WAT": 3,
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
    }
    cm_names = ["IS", "VEG", "SOIL", "WAT"]
    epochs = 100

    td.kw("CITY_NAME", city_name)
    td.kw("INIT_MODEL_NAME", init_model_name)
    td.kw("CSV_FN", csv_fn)
    td.kw("GET_NAMES", get_names)

    td.copyfile(csv_fn)
    td.copyfile(__file__)

    model = _GET_DL_MODEL(init_model_name, win_size, get_names)
    model_name = "{}_{}_{}-{}".format(city_name, init_model_name, win_size[0], win_size[1])
    dfn_tmp = DirFileName(td.fn(model_name))
    dfn_tmp.mkdir()

    read_size = (37, 37)

    td.log("\n#", "-" * 30, model_name, "-" * 30, "#\n")

    torch_mod = TorchModel()
    torch_mod.filename = dfn_tmp.fn(model_name + ".hm")
    torch_mod.map_dict = map_dict
    torch_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
    torch_mod.model = model
    torch_mod.criterion = nn.CrossEntropyLoss()
    torch_mod.win_size = win_size
    torch_mod.read_size = read_size
    torch_mod.epochs = epochs
    torch_mod.n_epoch_save = -1
    torch_mod.train_filters.extend([("city", "==", city_name)])
    torch_mod.test_filters.extend([("city", "==", city_name)])
    torch_mod.cm_names = cm_names

    td.kw("TORCH_MOD.FILENAME", torch_mod.filename)
    td.kw("TORCH_MOD.MAP_DICT", torch_mod.map_dict)
    td.kw("TORCH_MOD.COLOR_TABLE", torch_mod.color_table)
    td.kw("TORCH_MOD.MODEL", torch_mod.model.__class__)
    td.kw("TORCH_MOD.CRITERION", torch_mod.color_table)
    td.kw("TORCH_MOD.WIN_SIZE", torch_mod.win_size)
    td.kw("TORCH_MOD.READ_SIZE", torch_mod.read_size)
    td.kw("TORCH_MOD.EPOCHS", torch_mod.epochs)
    td.kw("TORCH_MOD.N_EPOCH_SAVE", torch_mod.n_epoch_save)
    td.kw("TORCH_MOD.CM_NAMES", torch_mod.cm_names)

    torch_mod.save_model_fmt = dfn_tmp.fn(model_name + "_" + city_name + "_{}.pth")
    td.kw("TORCH_MOD.SAVE_MODEL_FMT", torch_mod.save_model_fmt)

    def train():
        sd = SamplesData(_dl_sample_dirname=_DL_SAMPLE_DIRNAME)
        sd.addDLCSV(
            csv_fn, read_size, get_names, x_deal,
            grs={"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }
        )

        model_sw = td.buildWriteText(r"{}\{}.txt".format(model_name, model_name), "a")
        model_sw.write(torch_mod.model)

        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts(td.log)

        line_sw = td.buildWriteText(r"{}\{}_training-log.txt".format(model_name, model_name), "a")
        to_list = []

        def func_field_record_save(field_records):
            line = field_records.line
            to_list.append(line.copy())

            if int(line["Accuracy"]) != -1:
                for k in line:
                    line_sw.write("| {}:{} ".format(k, line[k]), end="")
                line_sw.write("|")
                if line["Batch"] == 0:
                    td.log("+ Epoch:", "{:<6d}".format(line["Epoch"]), end=" ", is_print=False)
                    td.log("Loss:", "{:<12.6f}".format(line["Loss"]), end=" ", is_print=False)
                    td.log("Accuracy:", "{:>6.3f}".format(line["Accuracy"]), end="\n", is_print=False)

        torch_mod.func_field_record_save = func_field_record_save
        torch_mod.train()
        td.saveJson(r"{}\{}_training-log.json".format(model_name, model_name), to_list)

    def imdc():
        torch_mod.x_keys = get_names
        torch_mod.imdc(
            raster_fn, data_deal=x_deal,
            mod_fn=torch_mod.save_model_fmt.format(epochs), read_size=(500, -1), is_save_tiles=True, fun_print=td.log
        )

    if is_train:
        train()
    else:
        imdc()

    return


class SHH2DLRun_TIC:

    def __init__(self, city_name, csv_fn, td=None,
                 map_dict=None, cm_names=None, x_keys=None,
                 color_table=None, x_deal=None,
                 ):
        self.city_name = city_name
        self.csv_fn = csv_fn
        self.raster_fn = SHH2Config.GET_RASTER_FN(self.city_name)
        self.range_fn = SHH2Config.GET_RANGE_FN(self.city_name)
        self.td = td
        self.initTD(td)

        self.x_keys = x_keys if x_keys is not None else _X_KEYS
        self.map_dict = map_dict if map_dict is not None else _MAP_DICT_4
        self.cm_names = cm_names if cm_names is not None else _CM_NAME_4
        self.color_table = color_table if color_table is not None else _COLOR_TABLE_4
        self.x_deal = x_deal if x_deal is not None else X_DEAL_18

        self.init_model_name = None
        self.model_name = None
        self.win_size = None
        self.model = None
        self.read_size = None
        self.models = {}
        self.models_records = {}
        self.accuracy_dict = {}

        self.sd = None
        self.torch_mod: TorchModel = TorchModel()
        self.dfn_tmp = DirFileName(_BACK_DIRNAME)
        self.epochs = None

        self.log("#", "-" * 50, self.city_name.upper(), "SHH2DL", "-" * 50, "#\n")
        self.kw("CITY_NAME", self.city_name)
        self.kw("CSV_FN", self.csv_fn)
        self.kw("RASTER_FN", self.raster_fn)
        self.kw("RANGE_FN", self.range_fn)
        self.kw("X_KEYS", self.x_keys)
        self.kw("MAP_DICT", self.map_dict)
        self.kw("CM_NAMES", self.cm_names)

    def initTD(self, td):
        if td is None:
            return
        self.td = td
        self.log(self.td.time_dfn.dirname)
        self.td.copyfile(__file__)

    def initModel(self, init_model_name, win_size):
        n_category = len(np.unique(self.cm_names))
        self.init_model_name = init_model_name
        self.model_name = "{}-{}-{}_{}".format(self.city_name.upper(), init_model_name, win_size[0], win_size[1])
        self.win_size = win_size
        self.model = _GET_DL_MODEL(init_model_name, win_size, len(self.x_keys), n_category)
        self.kw("INIT_MODEL_NAME", init_model_name)
        self.kw("MODEL_NAME", self.model_name)
        self.kw("WIN_SIZE", win_size)
        if self.td is not None:
            self.dfn_tmp = DirFileName(self.td.fn(self.model_name))
            self.dfn_tmp.mkdir()

    def initTorchModel(self, read_size, epochs=100, n_epoch_save=-1,
                       criterion=None, test_filters=None, train_filters=None, batch_size=32):

        if train_filters is None:
            train_filters = []
        if test_filters is None:
            test_filters = []

        self.read_size = read_size
        self.epochs = epochs

        torch_mod = TorchModel()
        torch_mod.filename = self.dfn_tmp.fn(self.model_name + ".hdlm")
        torch_mod.map_dict = self.map_dict
        torch_mod.color_table = self.color_table
        torch_mod.model = self.model
        torch_mod.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        torch_mod.win_size = self.win_size
        torch_mod.read_size = read_size
        torch_mod.epochs = epochs
        torch_mod.batch_size = batch_size
        torch_mod.n_epoch_save = n_epoch_save
        torch_mod.train_filters.extend([("city", "==", self.city_name), *train_filters])
        torch_mod.test_filters.extend([("city", "==", self.city_name), *test_filters])
        torch_mod.cm_names = self.cm_names
        torch_mod.save_model_fmt = self.dfn_tmp.fn(self.model_name + "_{}.pth")
        self.kw("TORCH_MOD.FILENAME", torch_mod.filename)
        self.kw("TORCH_MOD.MAP_DICT", torch_mod.map_dict)
        self.kw("TORCH_MOD.COLOR_TABLE", torch_mod.color_table)
        self.kw("TORCH_MOD.MODEL", torch_mod.model.__class__)
        self.kw("TORCH_MOD.CRITERION", torch_mod.color_table)
        self.kw("TORCH_MOD.WIN_SIZE", torch_mod.win_size)
        self.kw("TORCH_MOD.READ_SIZE", torch_mod.read_size)
        self.kw("TORCH_MOD.EPOCHS", torch_mod.epochs)
        self.kw("TORCH_MOD.N_EPOCH_SAVE", torch_mod.n_epoch_save)
        self.kw("TORCH_MOD.CM_NAMES", torch_mod.cm_names)
        self.kw("TORCH_MOD.SAVE_MODEL_FMT", torch_mod.save_model_fmt)

        self.torch_mod = torch_mod
        return torch_mod

    def initSD(self, csv_fn=None, read_size=None):
        if csv_fn is None:
            csv_fn = self.csv_fn
        if read_size is None:
            read_size = self.read_size
        csv_fn = os.path.abspath(csv_fn)
        self.csv_fn = csv_fn
        if self.td is not None:
            self.td.copyfile(self.csv_fn)
        self.sd = SamplesData(_dl_sample_dirname=_DL_SAMPLE_DIRNAME)
        self.sd.addDLCSV(
            csv_fn, read_size, self.x_keys, self.x_deal,
            grs={"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }
        )
        return

    def train(self):
        self.log("\n#", "-" * 50, self.model_name, "Training", "-" * 50, "#\n")

        self.initSD(self.csv_fn, self.read_size)

        torch_mod = self.torch_mod

        if self.td is not None:
            model_sw = SRTWriteText(self.dfn_tmp.fn("MODEL-{}.txt".format(self.model_name)), "a")
            model_sw.write(torch_mod.model)
            line_sw = SRTWriteText(self.dfn_tmp.fn(r"TRAINING-LOG-{}.txt".format(self.model_name)), "a")

        torch_mod.sampleData(self.sd)
        torch_mod.samples.showCounts(self.log)

        to_list = []

        def func_field_record_save(field_records):
            line = field_records.line
            to_list.append(line.copy())

            if int(line["Accuracy"]) != -1:
                if self.td is not None:
                    for k in line:
                        line_sw.write("| {}:{} ".format(k, line[k]), end="")
                line_sw.write("|")
                if line["Batch"] == 0:
                    self.log("+ Epoch:", "{:<6d}".format(line["Epoch"]), end=" ", is_print=False)
                    self.log("Loss:", "{:<12.6f}".format(line["Loss"]), end=" ", is_print=False)
                    self.log("Accuracy:", "{:>6.3f}".format(line["Accuracy"]), end="\n", is_print=False)

        torch_mod.func_field_record_save = func_field_record_save

        torch_mod.train()

        if self.td is not None:
            saveJson(to_list, self.dfn_tmp.fn(r"TRAINING-LOG-{}.json".format(self.model_name)))

    def imdc(self, raster_fn=None, mod_fn=None, raster_read_size=(500, -1)):
        self.log("\n#", "-" * 50, self.model_name, "Image Classification", "-" * 50, "#\n")

        if raster_fn is None:
            raster_fn = self.raster_fn
        if mod_fn is None:
            mod_fn = self.torch_mod.save_model_fmt.format(self.epochs)

        self.kw("RASTER_FN", self.raster_fn)

        self.torch_mod.x_keys = self.x_keys
        self.torch_mod.imdc(
            raster_fn, data_deal=self.x_deal,
            mod_fn=mod_fn, read_size=raster_read_size, is_save_tiles=True,
            fun_print=self.log
        )

    def kw(self, key, value, sep=": ", end="\n", is_print=None):
        if self.td is None:
            print(key, value, sep=sep, end=end, )
            return value
        else:
            return self.td.kw(key, value, sep=sep, end=end, is_print=is_print)

    def log(self, *text, sep=" ", end="\n", is_print=None):
        if self.td is None:
            print(*text, sep=sep, end=end, )
        else:
            self.td.log(*text, sep=sep, end=end, is_print=is_print)


def modelT(init_model_name, win_size):
    n_category = 4
    model = _GET_DL_MODEL(init_model_name, )
    x = torch.rand(2, in_ch, *win_size)
    out_x = model(x)
    print(init_model_name, x.shape, out_x.shape)


def updateISOToVHL(iso_imdc_fn, vhl_imdc_fn, to_imdc_fn):
    gr_iso = GDALRaster(iso_imdc_fn)
    gr_vhl = GDALRaster(vhl_imdc_fn)
    data_iso = gr_iso.readAsArray()
    data_vhl = gr_vhl.readAsArray()
    data_iso[data_iso == 2] = 3
    data_vhl[data_vhl == 3] = 4
    data_vhl[data_vhl == 1] = data_iso[data_vhl == 1]
    saveGTIFFImdc(gr_vhl, data_vhl, to_imdc_fn, _COLOR_TABLE_4)


def run(city_name, init_model_name, win_size, is_train, mod_fn=None):
    # td = TimeDirectory(_DL_DFN.fn(), time_dirname=W2LF(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240809H221443"))
    # td = TimeDirectory(_DL_DFN.fn(), time_dirname=W2LF(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240810H210858"))
    # td = TimeDirectory(_DL_DFN.fn(), time_dirname=W2LF(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240812H150307"))

    td = TimeDirectory(_DL_DFN.fn(), time_dirname=W2LF(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240820H110425"))
    td.initLog(mode="a")
    td.kw("DIRNAME", td.time_dirname())

    csv_fn = getCSVFn(city_name)
    read_size = (37, 37)

    tic = SHH2DLRun_TIC(
        city_name, csv_fn, td=td,

        map_dict={"IS": 0, "SOIL": 1},
        cm_names=["IS", "SOIL"],
        color_table={1: (255, 0, 0), 2: (255, 255, 0)},

    )
    tic.initModel(init_model_name, win_size)
    tic.initTorchModel(read_size, train_filters=[("FCNAME", "==", "ISO")])
    if is_train:
        tic.train()
    else:
        tic.imdc(mod_fn=mod_fn, raster_read_size=(1000, 800))


def running():
    def func1():
        run("qd", "VIT", (28, 28), False,
            r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240809H221443\qd_VIT_28-28\qd_VIT_28-28_qd_100.pth")

    def func2():
        run("qd", "ResNet18", (7, 7), True)

    return func2()


def main():
    win_size_list = [
        ("ResNet18", (7, 7)),
        ("ResNeXt5032x4d", (7, 7)),
        ("DenseNet121", (28, 28)),
        # ("SqueezeNet", (24, 24)),
        ("GoogLeNet", (28, 28)),
        ("Inception3", (35, 35)),  # size small
        # ("ShuffleNetV2X05", (7, 7)),  # 轻量化
        # ("MobileNetV2", (7, 7)),  # 轻量化
        # ("MobileNetV3Small", (7, 7)),  # 轻量化
        # ("EfficientNetB7", (7, 7)), # 单独弄，因为它主打一个在图片分类精度最高，Google的一个网络，Google出了好多网络
        # ("EfficientNetV2M", (7, 7)),
        ("VIT", (28, 28)),
    ]

    # run_list = RunList_V2(r"F:\Week\20240818\Data\RunList_V2.json")
    #
    # def init():
    #     run_list.add(name="RunList", number=1, )
    #     run_list.add(name="RunList", number=2, )
    #     run_list.add(name="RunListV2", number=1, )
    #     run_list.add(name="RunListV2", number=2, )
    #     run_list.global_args["_DIR_NAME"] = r"F:\Week\20240818\Data"
    #     run_list.show()
    #     run_list.saveToJson()
    #
    # def func51(name, number, _DIR_NAME):
    #     print("_DIR_NAME:", _DIR_NAME)
    #     print("Name:", name)
    #     print("Number:", number)
    #     time.sleep(1)
    #     return {"_DIR_NAME": os.path.join(_DIR_NAME, name, str(number))}
    #
    # init()

    # run_list.fit(func51).saveToJson()

    # for name, win_size in win_size_list:
    #     modelT(name, win_size)
    #
    # sys.exit()

    for city_name in ["qd", "bj", "cd"]:
        for name, win_size in win_size_list:
            print("python -c \""
                  r"import sys; "
                  r"sys.path.append(r'F:\PyCodes'); "
                  r"from Shadow.Hierarchical.SHH2DLRun import run; "
                  "run('{}', '{}', {}, True)\"".format(
                city_name, name, win_size
            ))
            print("python -c \""
                  r"import sys; "
                  r"sys.path.append(r'F:\PyCodes'); "
                  r"from Shadow.Hierarchical.SHH2DLRun import run; "
                  "run('{}', '{}', {}, False)\"".format(
                city_name, name, win_size
            ))
        print("\n")


if __name__ == "__main__":
    main()
    # run2('qd', 'ResNet18', (7, 7), True)
    # updateISOToVHL(
    #     iso_imdc_fn=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240810H210858\QD-ResNeXt5032x4d-7_7\QD-ResNeXt5032x4d-7_7_100_imdc.tif",
    #     vhl_imdc_fn=r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240812H144454\VHL-QD-O_imdc.tif",
    #     to_imdc_fn=r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240810H210858\QD-ResNeXt5032x4d-7_7\QD-ResNeXt5032x4d-7_7_100_upateVHL_imdc.tif"
    # )
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DLRun import main; main()"

python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DLRun import run2; run2('qd', 'ResNeXt5032x4d', (7, 7), True)"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DLRun import run2; run2('qd', 'ResNeXt5032x4d', (7, 7), False)"


    """
