# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowASDESamplesDL.py
@Time    : 2025/4/21 13:23
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2025, ZhengHan. All rights reserved.
@Desc    : PyCodes of HsoadDl
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
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALTorch import GDALTorchImdc
from SRTCodes.SRTModel import SamplesData, TorchModel
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import DirFileName, SRTWriteText, saveJson, mkdir

_QD_SPL_FN = r"F:\ASDEWrite\Run\Deeplearning\qd_hsoad_spl.csv"
_BJ_SPL_FN = r"F:\ASDEWrite\Run\Deeplearning\bj_hsoad_spl.csv"
_CD_SPL_FN = r"F:\ASDEWrite\Run\Deeplearning\cd_hsoad_spl.csv"

_QD_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\QingDao\HSPL_QD_envi.dat"
_BJ_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\BeiJing\HSPL_BJ_envi.dat"
_CD_RASTER_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\ChengDu\HSPL_CD_envi.dat"

_QD_RANGE_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\QingDao\HSPL_QD.range"
_BJ_RANGE_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\BeiJing\HSPL_BJ.range"
_CD_RANGE_FN = r"F:\ProjectSet\Shadow\ASDEHSamples\Images\ChengDu\HSPL_CD.range"

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
_BACK_DIRNAME = r"F:\ASDEWrite\Run\Models\Temp"
_DL_SAMPLE_DIRNAME = r"F:\ASDEWrite\Run\Deeplearning"
_DL_DFN = DirFileName(r"F:\ASDEWrite\Run\Models")


def _GET_CITY_NAME(city_name, qd, bj, cd):
    if city_name == "qd":
        return qd
    elif city_name == "bj":
        return bj
    elif city_name == "cd":
        return cd
    else:
        raise Exception("Can not find \"{}\"".format(city_name))


def _GET_RASTER_FN(city_name):
    return _GET_CITY_NAME(city_name, _QD_RASTER_FN, _BJ_RASTER_FN, _CD_RASTER_FN)


def _GET_RANGE_FN(city_name):
    return _GET_CITY_NAME(city_name, _QD_RANGE_FN, _BJ_RANGE_FN, _CD_RANGE_FN)


def X_DEAL_18(_x):
    for i in range(0, 6):
        _x[i] = _x[i] / 160
    # for i in range(6, 10):
    #     _x[i] = (_x[i] + 30) / 35
    # for i in range(12, 16):
    #     _x[i] = (_x[i] + 30) / 35
    # _x[11] = _x[11] / 90
    # _x[17] = _x[11] / 90
    return _x


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


class SHDL_TIC:

    def __init__(self, city_name, csv_fn, td=None,
                 map_dict=None, cm_names=None, x_keys=None,
                 color_table=None, x_deal=None,
                 ):
        self.city_name = city_name
        self.csv_fn = csv_fn
        self.raster_fn = _GET_RASTER_FN(self.city_name)
        self.range_fn = _GET_RANGE_FN(self.city_name)
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

        self.log("#", "-" * 50, self.city_name.upper(), "SHDL", "-" * 50, "#\n")
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
        torch_mod.filename = self.dfn_tmp.fn(self.model_name + ".d")
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
        torch_mod.save_model_end_fn = self.dfn_tmp.fn(self.model_name + "_tmp.pth")
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
            grs={"qd": GDALRaster(_QD_RASTER_FN), "bj": GDALRaster(_BJ_RASTER_FN), "cd": GDALRaster(_CD_RASTER_FN), },
            is_remove_no_spl=True
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

        torch_mod.train(lr=0.0006)

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


def run(city_name, init_model_name, win_size, is_train, mod_fn=None):
    time_dirname = mkdir("F:\\ASDEWrite\\Run\\Models\\20250421H150823\\{}".format(city_name))
    color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
    x_deal = X_DEAL_18
    epochs = 100
    cm_names = ["IS", "VEG", "SOIL", "WAT"]
    x_keys = [
        "Blue", "Green", "Red", "NIR",
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ]

    if is_train:
        td = TimeDirectory(_DL_DFN.fn(), time_dirname=time_dirname)
        td.initLog()
        tic = SHDL_TIC(
            city_name, _GET_CITY_NAME(city_name, _QD_SPL_FN, _BJ_SPL_FN, _CD_SPL_FN, ), td=td,
            map_dict={
                "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
                "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
                "IS_AS_SH": 0, "VEG_AS_SH": 1, "SOIL_AS_SH": 2, "WAT_AS_SH": 3,
                "IS_DE_SH": 0, "VEG_DE_SH": 1, "SOIL_DE_SH": 2, "WAT_DE_SH": 3,
            }, cm_names=cm_names, x_keys=x_keys,
            color_table=color_table,
            x_deal=x_deal,
        )

        tic.initModel(init_model_name, win_size)
        tic.initTorchModel(read_size=[37, 37], epochs=epochs)

        tic.train()
    else:
        model_name = "{}-{}-{}_{}".format(city_name.upper(), init_model_name, win_size[0], win_size[1])
        mod_fn = os.path.join(time_dirname, model_name, "{}_{}.pth".format(model_name, epochs),)
        if not os.path.isfile(mod_fn):
            raise Exception("Can not find model file: {}".format(mod_fn))

        to_imdc_fn = os.path.join(time_dirname, model_name, "{}_{}_imdc.tif".format(model_name, epochs))
        n_category = len(np.unique(cm_names))
        model = _GET_DL_MODEL(init_model_name, win_size, len(x_keys), n_category)
        model.load_state_dict(torch.load(mod_fn))
        model.to("cuda")
        model.zero_grad()
        model.eval()

        sw = SRTWriteText(
            os.path.join(time_dirname, model_name, "{}_{}_logimdc.txt".format(model_name, epochs)),
            is_show=True,
        )
        sw.write("MODEL_NAME:", model_name)
        sw.write("TO_IMDC_FN:", to_imdc_fn)
        sw.write("MODEL:", model)

        def func_predict(x):
            with torch.no_grad():
                logit = model(x)
                y = torch.argmax(logit, dim=1) + 1
            return y

        gti = GDALTorchImdc(_GET_RASTER_FN(city_name))
        gti.imdc3(
            func_predict=func_predict, win_size=win_size, to_imdc_fn=to_imdc_fn,
            fit_names=x_keys, data_deal=x_deal, color_table=color_table,
            is_jdt=True, device="cuda", read_size=(-1, 1000), is_save_tiles=True,
            fun_print=sw.write,
        )


def main():
    def func1():
        td = TimeDirectory(r"F:\ASDEWrite\Run\Models")
        td.initLog()
        print(td.time_dirname())

        cm_names = ["IS", "VEG", "SOIL", "WAT"]
        get_names = [
            "Blue", "Green", "Red", "NIR",
            "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
            "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
        ]
        csv_fn = _QD_SPL_FN
        grs = {"qd": GDALRaster(_QD_RASTER_FN), "bj": GDALRaster(_QD_RASTER_FN), "cd": GDALRaster(_QD_RASTER_FN), }
        sd = SamplesData(_dl_sample_dirname=r"F:\ASDEWrite\Run\Deeplearning")
        sd.addDLCSV(csv_fn, (21, 21), get_names, grs=grs)

        torch_mod = TorchModel()
        torch_mod.filename = None
        torch_mod.map_dict = {
            "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
            "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
            "IS_AS_SH": 0, "VEG_AS_SH": 1, "SOIL_AS_SH": 2, "WAT_AS_SH": 3,
            "IS_DE_SH": 0, "VEG_DE_SH": 1, "SOIL_DE_SH": 2, "WAT_DE_SH": 3,
        }
        torch_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        model = nn.Sequential(
            nn.Conv2d(len(get_names), len(get_names), 3, 1, 1),
            nn.Flatten(start_dim=1),
            nn.Linear(21 * 21 * len(get_names), 4),
        )
        # model = buildModel(None)
        torch_mod.model = model
        torch_mod.criterion = nn.CrossEntropyLoss()
        torch_mod.win_size = (21, 21)
        torch_mod.read_size = (21, 21)
        torch_mod.epochs = 10
        torch_mod.cm_names = cm_names
        # torch_mod.train_filters.append(("city", "==", "cd"))
        # torch_mod.test_filters.append(("city", "==", "cd"))
        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts()
        torch_mod.save_model_fmt = td.time_dirname() + "\\model{}.pth"
        torch_mod.train(lr=0.0005)
        torch_mod.imdc(_QD_RASTER_FN, mod_fn=torch_mod.save_model_fmt.format(torch_mod.epochs))

        # mod_fn = None
        # if mod_fn is not None:
        #     torch_mod.imdc([
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_1.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_2.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_3.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_4.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_1.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_2.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_3.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_4.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_1.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_2.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_3.tif",
        #         r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_4.tif",
        #     ], mod_fn=mod_fn)

    def func2():
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
        td = TimeDirectory(_DL_DFN.fn())
        td.initLog()

        tic = SHDL_TIC(
            "qd", _QD_SPL_FN, td=td,
            map_dict={
                "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
                "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
                "IS_AS_SH": 0, "VEG_AS_SH": 1, "SOIL_AS_SH": 2, "WAT_AS_SH": 3,
                "IS_DE_SH": 0, "VEG_DE_SH": 1, "SOIL_DE_SH": 2, "WAT_DE_SH": 3,
            }, cm_names=["IS", "VEG", "SOIL", "WAT"], x_keys=[
                "Blue", "Green", "Red", "NIR",
                "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
                "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
            ],
            color_table={1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), },
            x_deal=X_DEAL_18,
        )

        tic.initModel("ResNet18", (7, 7))
        tic.initTorchModel(read_size=[37, 37], epochs=3)

        tic.train()
        tic.imdc(raster_read_size=(-1, -1))

    def func3():
        win_size_list = [
            ("ResNet18", (7, 7)),
            ("ResNeXt5032x4d", (7, 7)),
            ("DenseNet121", (28, 28)),
            # ("SqueezeNet", (24, 24)),
            ("GoogLeNet", (28, 28)),
            # ("Inception3", (35, 35)),  # size small
            # ("ShuffleNetV2X05", (7, 7)),  # 轻量化
            # ("MobileNetV2", (7, 7)),  # 轻量化
            # ("MobileNetV3Small", (7, 7)),  # 轻量化
            # ("EfficientNetB7", (7, 7)), # 单独弄，因为它主打一个在图片分类精度最高，Google的一个网络，Google出了好多网络
            # ("EfficientNetV2M", (7, 7)),
            ("VIT", (28, 28)),
        ]
        for name, win_size in win_size_list:
            for city_name in ["qd", "bj", "cd"]:
                print("python -c \""
                      r"import sys; "
                      r"sys.path.append(r'F:\PyCodes'); "
                      r"from Shadow.ShadowASDESamplesDL import run; "
                      "run('{}', '{}', {}, True)\"".format(
                    city_name, name, win_size
                ))
                print("python -c \""
                      r"import sys; "
                      r"sys.path.append(r'F:\PyCodes'); "
                      r"from Shadow.ShadowASDESamplesDL import run; "
                      "run('{}', '{}', {}, False)\"".format(
                    city_name, name, win_size
                ))
            print("\n")

    return func3()


if __name__ == "__main__":
    main()

