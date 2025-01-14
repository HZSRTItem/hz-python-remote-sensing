# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2DLIntersect.py
@Time    : 2024/8/13 9:52
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2DLIntersect
-----------------------------------------------------------------------------"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate
from torch import nn, optim
from torch.utils.data import Dataset

from SRTCodes.GDALRasterIO import GDALRaster, saveGTIFFImdc
from SRTCodes.ModelTraining import dataModelPredict
from SRTCodes.PytorchModelTraining import TorchTraining
from SRTCodes.SRTFeature import SRTFeaturesCalculation
from SRTCodes.SRTModel import mapDict
from SRTCodes.Utils import changext, readJson, DirFileName, saveJson, SRTWriteText
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Sample import SAMPLING_CITY_NAME

_F = SHH2Config.FEAT_NAMES
_MAP_DICT = {"IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4}
_COLOR_TABLE_4 = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }


def _GET_DF(city_name, csv_fn):
    to_spl_csv_fn = changext(csv_fn, "_spl.csv")
    if not os.path.isfile(to_spl_csv_fn):
        SAMPLING_CITY_NAME(city_name, csv_fn, to_spl_csv_fn)
    return pd.read_csv(to_spl_csv_fn)


def _DATA_RANGE(data, range_dict_0):
    data = np.clip(data, range_dict_0["min"], range_dict_0["max"])
    data = (data - range_dict_0["min"]) / (range_dict_0["max"] - range_dict_0["min"])
    return data


def dataRange(df, range_dict, sfc: SRTFeaturesCalculation):
    sfc.init_names = []
    for name in range_dict:
        df[name] = _DATA_RANGE(df[name].values, range_dict[name])
        sfc.init_names.append(name)
        sfc.add(name, [name], lambda _dict: _DATA_RANGE(_dict[name], range_dict[name]))
    return df


def _FUNC_LOGIT_CATEGORY(model, x: torch.Tensor):
    with torch.no_grad():
        # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1182 {} >".format(x.shape))
        logit = model(x)
        # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1184>")
        y = torch.argmax(logit, dim=1) + 1
    return y


class DLM_ReduceLogits(nn.Module):

    def __init__(self, opt_x_keys, oad_x_keys, find_x_keys, n_category):
        super(DLM_ReduceLogits, self).__init__()

        self.opt_x_keys = opt_x_keys
        self.oad_x_keys = oad_x_keys
        self.find_x_keys = find_x_keys

        self.opt_x_list = [self.find_x_keys.index(name) for name in self.opt_x_keys]
        self.oad_x_list = [self.find_x_keys.index(name) for name in self.oad_x_keys]

        in_ch_opt, in_ch_oad = len(opt_x_keys), len(oad_x_keys)
        self.opt = nn.Linear(in_ch_opt, n_category)
        self.oad = nn.Linear(in_ch_oad, n_category)

        self.is_imdc = False

    def forward(self, x):
        x_opt = x[:, self.opt_x_list]
        x_opt = self.opt(x_opt)

        x_oad = x[:, self.oad_x_list]
        x_oad = self.oad(x_oad)

        if self.is_imdc:
            return x_oad

        return x_opt, x_oad


class DLL_ReduceLogits(nn.Module):

    def __init__(self, n_change_loss=100):
        super(DLL_ReduceLogits, self).__init__()
        self.opt_cross_entropy_loss = nn.CrossEntropyLoss()
        self.oad_cross_entropy_loss = nn.CrossEntropyLoss()
        self.n_change_loss = n_change_loss
        self._n_change_loss_current = 0

    def forward(self, logits, y):
        logits_opt, logits_oad = logits

        if self._n_change_loss_current >= self.n_change_loss:
            # loss_opt = self.opt_cross_entropy_loss(logits_opt, y)
            loss_oad = self.oad_cross_entropy_loss(logits_oad, y)
            loss_reduce = torch.mean(torch.pow(torch.sigmoid(logits_opt) - torch.sigmoid(logits_oad), 2))
            return loss_oad + loss_reduce
        else:
            loss_opt = self.opt_cross_entropy_loss(logits_opt, y)
            loss_oad = self.oad_cross_entropy_loss(logits_oad, y)
            return loss_opt, loss_oad


def training(model, xy, test, epochs, cm_names, epoch_save, save_model_fmt,
             func_logit_category=_FUNC_LOGIT_CATEGORY, criterion=None, func_print=print,
             func_wights_save=None
             ):
    x, y = xy

    class ds(Dataset):

        def __init__(self):
            self.x_list = []
            self.y_list = []

        def __getitem__(self, item):
            return self.x_list[item], self.y_list[item]

        def __len__(self):
            return len(self.x_list)

    train_ds, test_ds = ds(), ds()
    for i in range(len(x)):
        if test[i] == 1:
            train_ds.x_list.append(x[i])
            train_ds.y_list.append(y[i])
        else:
            test_ds.x_list.append(x[i])
            test_ds.y_list.append(y[i])

    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    batch_size = 32

    torch_training = TorchTraining()

    torch_training.model = model
    torch_training.criterion = criterion
    torch_training.epochs = epochs
    torch_training.device = "cpu"
    torch_training.n_test = 10

    torch_training.trainLoader(train_ds, batch_size=32)
    torch_training.testLoader(test_ds, batch_size=batch_size)

    torch_training.optimizer(optim.Adam, lr=0.001, eps=0.000001)
    torch_training.scheduler(optim.lr_scheduler.StepLR, step_size=30, gamma=0.76, last_epoch=-1)

    torch_training.initCM(cnames=cm_names)
    torch_training.batch_save = False
    torch_training.epoch_save = epoch_save
    torch_training.save_model_fmt = save_model_fmt
    torch_training.n_epoch_save = -1

    to_list = []

    def func_field_record_save(field_records):
        line = field_records.line
        if int(line["Accuracy"]) != -1:
            if line["Batch"] == 0:
                to_list.append(line.copy())
                func_print("+ Epoch:", "{:<6d}".format(line["Epoch"]), end=" ", )
                func_print("Loss:", "{:<12.6f}".format(line["Loss"]), end=" ", )
                func_print("Accuracy:", "{:>6.3f}".format(line["Accuracy"]), end="\n", )
                if func_wights_save is not None:
                    func_wights_save(torch_training)

    def func_batch():
        torch_training.criterion._n_change_loss_current = torch_training.epoch

    # torch_training.func_epoch = self.func_epoch
    # torch_training.func_xy_deal = self.func_xy_deal
    torch_training.func_batch = func_batch
    # torch_training.func_loss_deal = self.func_loss_deal
    # torch_training.func_y_deal = self.func_y_deal
    torch_training.func_logit_category = func_logit_category
    # torch_training.func_print = func_print
    torch_training.func_field_record_save = func_field_record_save

    torch_training.train()

    return torch_training, to_list


def imdc(model, raster_fn, to_imdc_fn, x_keys, range_dict,
         _func_logit_category=_FUNC_LOGIT_CATEGORY, color_table=None):
    if color_table is None:
        color_table = _COLOR_TABLE_4

    gr = GDALRaster(raster_fn)
    data = gr.readGDALBands(*x_keys)
    for i, name in enumerate(x_keys):
        data[i] = _DATA_RANGE(data[i], range_dict[name])

    class this_model_cls:

        def predict(self, x):
            return _func_logit_category(model, x).numpy()

    this_model = this_model_cls()
    data = torch.from_numpy(data)
    to_data = dataModelPredict(data, data_deal=None, is_jdt=True, model=this_model)
    saveGTIFFImdc(gr, to_data, to_imdc_fn, color_table=color_table)


def getXY(df, x_keys, map_dict, func_print=print):
    x = df[x_keys].values
    x = x.astype("float32")
    y = mapDict(df["CNAME"].to_list(), map_dict)
    data = np.concatenate([np.array([y]).T, x], axis=1)
    func_print(tabulate(data[:6, :], headers=["CATEGORY", *x_keys], tablefmt="simple"))
    return x, y


def saveLinesToCSV(to_csv_fn, lines, ):
    with open(to_csv_fn, "w", encoding="utf-8", newline="") as f:
        cw = csv.writer(f)
        for line in lines:
            cw.writerow(line)


def show():
    _DFN = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240813H104039")

    def func1():
        json_dict = readJson(_DFN.fn(r"7\qd-ReduceLogits_training.json"))
        df = pd.DataFrame(json_dict)
        print(df.keys())
        plt.plot(df["OA Test"])
        plt.show()

    func1()


def showWight(wights, _x_keys, cm_names, sw):
    wights = wights.numpy().T.tolist()
    to_list = [["X_KEYS", *cm_names], *[[_x_keys[i]] + wights[i] for i in range(len(_x_keys))]]
    sw.write("# WIGHTS ------")
    sw.write(tabulate(to_list, headers="firstrow", tablefmt="simple"))
    return to_list


def numberDirname(dirname):
    for i in range(1, 1000):
        to_dirname = os.path.join(dirname, str(i))
        if not os.path.isdir(to_dirname):
            return to_dirname
    raise Exception("Can not number dirname {}".format(dirname))


def init(city_name, csv_fn):
    range_dict = readJson(SHH2Config.GET_RANGE_FN(city_name))
    raster_fn = SHH2Config.GET_RASTER_FN(city_name)
    to_dfn = DirFileName(numberDirname(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240813H104039"))
    to_dfn.mkdir()
    to_dfn.copyfile(__file__)
    sw = SRTWriteText(to_dfn.fn("log.txt"), is_show=True)
    df = _GET_DF(city_name, csv_fn)
    to_dfn.copyfile(changext(csv_fn, "_spl.csv"))
    sfc = SRTFeaturesCalculation()
    df = dataRange(df, range_dict, sfc)
    return range_dict, raster_fn, to_dfn, sw, df, sfc


def swWight(model, _x_keys, cm_names, sw, to_fn):
    """to_dfn.fn("{}-{}-wights.csv".format(city_name, _name))"""
    wights = model.weight.data
    to_list = showWight(wights, _x_keys, cm_names=cm_names, sw=sw)
    saveLinesToCSV(to_fn, to_list, )


class TIC:

    def __init__(self, city_name, csv_fn):
        self.city_name = city_name
        self.csv_fn = csv_fn
        range_dict, raster_fn, to_dfn, sw, df, sfc = init(city_name, csv_fn)
        self.range_dict = range_dict
        self.raster_fn = raster_fn
        self.to_dfn = to_dfn
        self.sw = sw
        self.df = df
        self.sfc = sfc

    def train1(self, model, _name, _x_keys, _map_dict, _cm_names, ):
        """ model = nn.Linear(len(_x_keys), len(_cm_names)) """
        self.sw.write("#", "-" * 30, _name, "-" * 30, "#")
        self.sw.write("X_KEYS:", _x_keys)
        self.sw.write("MAP_DICT:", _map_dict)
        self.sw.write("CM_NAMES:", _cm_names)

        to_mod_fn = self.to_dfn.fn("{}-{}.pth".format(self.city_name, _name))
        torch_training_opt, to_list = training(
            model=model,
            xy=getXY(self.df, _x_keys, _map_dict, func_print=self.sw.write),
            test=self.df["TEST"].to_list(),
            epochs=100,
            cm_names=_cm_names,
            epoch_save=False,
            save_model_fmt=None,
            func_print=self.sw.write
        )
        torch.save(torch_training_opt.model.state_dict(), to_mod_fn)

        saveJson(to_list, self.to_dfn.fn("{}-{}_training.json".format(self.city_name, _name)))

    def imdc1(self, model, _x_keys, _name):
        """ model.load_state_dict(torch.load(to_mod_fn)) """

        imdc(
            model=model,
            raster_fn=self.raster_fn,
            to_imdc_fn=self.to_dfn.fn("{}-{}_imdc.tif".format(self.city_name, _name)),
            x_keys=_x_keys, range_dict=self.range_dict,
            _func_logit_category=_FUNC_LOGIT_CATEGORY, color_table=_COLOR_TABLE_4,
        )

    def train2(self,_name, _opt_x_keys, _oad_x_keys, _map_dict, _cm_names):

        _find_x_keys = list(set(_opt_x_keys + _oad_x_keys))

        self.sw.write("#", "-" * 30, _name, "-" * 30, "#")
        self.sw.write("OPT_X_KEYS:", _opt_x_keys)
        self.sw.write("OAD_X_KEYS:", _oad_x_keys)
        self.sw.write("FIND_X_KEYS:", _oad_x_keys)
        self.sw.write("MAP_DICT:", _map_dict)
        self.sw.write("CM_NAMES:", _cm_names)

        to_wights_list = []

        def func_wights_save(_torch_training: TorchTraining):
            to_wights_list.append({
                "opt": _torch_training.model.opt.weight.data.numpy().T.tolist(),
                "oad": _torch_training.model.oad.weight.data.numpy().T.tolist(),
            })

        model = DLM_ReduceLogits(_opt_x_keys, _oad_x_keys, _find_x_keys, len(_cm_names))

        to_mod_fn = to_dfn.fn("{}-{}.pth".format(city_name, _name))

        def _func_logit_category(_model, x: torch.Tensor):
            _model.is_imdc = True
            with torch.no_grad():
                # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1182 {} >".format(x.shape))
                logit = _model(x)
                # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1184>")
                y = torch.argmax(logit, dim=1) + 1
            _model.is_imdc = False
            return y

        torch_training_opt, to_list = training(
            model=model,
            xy=getXY(df, _find_x_keys, _map_dict, func_print=sw.write),
            test=df["TEST"].to_list(),
            epochs=120, cm_names=_cm_names, epoch_save=False, save_model_fmt=None,
            criterion=DLL_ReduceLogits(n_change_loss=60),
            func_logit_category=_func_logit_category,
            func_print=self.sw.write, func_wights_save=func_wights_save
        )

        torch.save(torch_training_opt.model.state_dict(), to_mod_fn)
        saveJson(to_list, to_dfn.fn("{}-{}_training.json".format(city_name, _name)))
        to_list = showWight(model.opt.weight.data, _opt_x_keys, cm_names=cm_names, sw=sw)
        saveLinesToCSV(to_dfn.fn("{}-{}-opt-wights.csv".format(city_name, _name)), to_list, )
        to_list = showWight(model.oad.weight.data, _oad_x_keys, cm_names=cm_names, sw=sw)
        saveLinesToCSV(to_dfn.fn("{}-{}-oad-wights.csv".format(city_name, _name)), to_list, )
        saveJson(to_wights_list, to_dfn.fn("{}-{}-wights.json".format(city_name, _name)))

        model.is_imdc = True
        imdc(
            model=model,
            raster_fn=raster_fn,
            to_imdc_fn=to_dfn.fn("{}-{}_imdc.tif".format(city_name, _name)),
            x_keys=_find_x_keys, range_dict=range_dict,
            _func_logit_category=_func_logit_category, color_table=_COLOR_TABLE_4
        )


def main():
    city_name = "qd"
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd6.csv"
    range_dict, raster_fn, to_dfn, sw, df, sfc = init(city_name, csv_fn)

    def func1(_name, _x_keys, _map_dict, _cm_names):
        sw.write("#", "-" * 30, _name, "-" * 30, "#")
        sw.write("X_KEYS:", _x_keys)
        sw.write("MAP_DICT:", _map_dict)
        sw.write("CM_NAMES:", _cm_names)

        model = nn.Linear(len(_x_keys), len(_cm_names))

        to_mod_fn = to_dfn.fn("{}-{}.pth".format(city_name, _name))

        torch_training_opt, to_list = training(
            model=model,
            xy=getXY(df, _x_keys, _map_dict, func_print=sw.write),
            test=df["TEST"].to_list(),
            epochs=100, cm_names=_cm_names, epoch_save=False, save_model_fmt=None,
            func_print=sw.write
        )
        torch.save(torch_training_opt.model.state_dict(), to_mod_fn)
        saveJson(to_list, to_dfn.fn("{}-{}_training.json".format(city_name, _name)))

        model.load_state_dict(torch.load(to_mod_fn))
        imdc(
            model=model,
            raster_fn=raster_fn,
            to_imdc_fn=to_dfn.fn("{}-{}_imdc.tif".format(city_name, _name)),
            x_keys=_x_keys, range_dict=range_dict,
            _func_logit_category=_FUNC_LOGIT_CATEGORY, color_table=_COLOR_TABLE_4,
        )

        wights = model.weight.data
        to_list = showWight(wights, _x_keys, cm_names=cm_names, sw=sw)
        saveLinesToCSV(to_dfn.fn("{}-{}-wights.csv".format(city_name, _name)), to_list, )

    name = "opt"
    map_dict = {"IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3, "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3}
    cm_names = ["IS", "VEG", "SOIL", "WAT"]
    x_keys = _F.OPT + _F.OPT_GLCM

    # func1("o", _F.OPT + _F.OPT_GLCM, map_dict, cm_names)
    # func1("oa", _F.OPT + _F.OPT_GLCM + _F.AS, map_dict, cm_names)
    # func1("od", _F.OPT + _F.OPT_GLCM + _F.DE, map_dict, cm_names)
    # func1("oad", _F.ALL, map_dict, cm_names)

    # func1("o", _F.OPT, map_dict, cm_names)
    # func1("oa", _F.OPT + _F.AS, map_dict, cm_names)
    # func1("od", _F.OPT + _F.DE, map_dict, cm_names)
    # func1("oad", _F.OPT + _F.AS + _F.DE, map_dict, cm_names)

    def func2(_opt_x_keys, _oad_x_keys, _map_dict, _cm_names):
        _find_x_keys = list(set(_opt_x_keys + _oad_x_keys))
        _name = "ReduceLogits"

        sw.write("#", "-" * 30, _name, "-" * 30, "#")
        sw.write("OPT_X_KEYS:", _opt_x_keys)
        sw.write("OAD_X_KEYS:", _oad_x_keys)
        sw.write("FIND_X_KEYS:", _oad_x_keys)
        sw.write("MAP_DICT:", _map_dict)
        sw.write("CM_NAMES:", _cm_names)

        to_wights_list = []

        def func_wights_save(_torch_training: TorchTraining):
            to_wights_list.append({
                "opt": _torch_training.model.opt.weight.data.numpy().T.tolist(),
                "oad": _torch_training.model.oad.weight.data.numpy().T.tolist(),
            })

        model = DLM_ReduceLogits(_opt_x_keys, _oad_x_keys, _find_x_keys, len(_cm_names))
        to_mod_fn = to_dfn.fn("{}-{}.pth".format(city_name, _name))

        def _func_logit_category(_model, x: torch.Tensor):
            _model.is_imdc = True
            with torch.no_grad():
                # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1182 {} >".format(x.shape))
                logit = _model(x)
                # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1184>")
                y = torch.argmax(logit, dim=1) + 1
            _model.is_imdc = False
            return y

        torch_training_opt, to_list = training(
            model=model,
            xy=getXY(df, _find_x_keys, _map_dict, func_print=sw.write),
            test=df["TEST"].to_list(),
            epochs=120, cm_names=_cm_names, epoch_save=False, save_model_fmt=None,
            criterion=DLL_ReduceLogits(n_change_loss=60),
            func_logit_category=_func_logit_category,
            func_print=sw.write, func_wights_save=func_wights_save
        )

        torch.save(torch_training_opt.model.state_dict(), to_mod_fn)
        saveJson(to_list, to_dfn.fn("{}-{}_training.json".format(city_name, _name)))
        to_list = showWight(model.opt.weight.data, _opt_x_keys, cm_names=cm_names, sw=sw)
        saveLinesToCSV(to_dfn.fn("{}-{}-opt-wights.csv".format(city_name, _name)), to_list, )
        to_list = showWight(model.oad.weight.data, _oad_x_keys, cm_names=cm_names, sw=sw)
        saveLinesToCSV(to_dfn.fn("{}-{}-oad-wights.csv".format(city_name, _name)), to_list, )
        saveJson(to_wights_list, to_dfn.fn("{}-{}-wights.json".format(city_name, _name)))

        model.is_imdc = True
        imdc(
            model=model,
            raster_fn=raster_fn,
            to_imdc_fn=to_dfn.fn("{}-{}_imdc.tif".format(city_name, _name)),
            x_keys=_find_x_keys, range_dict=range_dict,
            _func_logit_category=_func_logit_category, color_table=_COLOR_TABLE_4
        )

    func2(_F.OPT, _F.ALL, map_dict, cm_names)


def mainISO():
    return


if __name__ == "__main__":
    main()
    r"""
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DLIntersect import main; main()"
    """
