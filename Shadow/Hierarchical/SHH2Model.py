# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Model.py
@Time    : 2024/7/4 14:13
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Model
-----------------------------------------------------------------------------"""

import os.path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch import nn

from SRTCodes.SRTModel import DataScale, _ModelInit, MLModel, _Samples, _MLSamples, _FieldData, _sampleTestCounts, \
    _Sample, TorchModel, _imdc1, _funcFilter
from SRTCodes.SRTModel import SamplesData as _SamplesData
from SRTCodes.Utils import saveJson, readJson
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2Config import FEAT_NAMES

_DL_SAMPLE_DIRNAME = r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL"

_DataScale = DataScale


class _MLFCSamples(_Samples):

    def __init__(self):
        super().__init__()

        self.vhl_spls = _MLSamples()
        self.is_spls = _MLSamples()
        self.ws_spls = _MLSamples()

        self.vhl_fd = _FieldData()
        self.is_fd = _FieldData()
        self.ws_fd = _FieldData()

    def deal(self):
        def _field_data_func(_spl: _MLSamples, _fd: _FieldData):
            _fd.dim = 1
            _spl.x_train = _fd.get(_spl.x_train)
            _spl.x_test = _fd.get(_spl.x_test)

        _field_data_func(self.vhl_spls, self.vhl_fd)
        _field_data_func(self.is_spls, self.is_fd)
        _field_data_func(self.ws_spls, self.ws_fd)

        self.x_test = self.data_scale.fits(pd.DataFrame(self.x_test))
        self.x_test = self.x_test.values
        return self

    def showCounts(self,func_print=print, *args, **kwargs):
        func_print("* VHL Sample Counts")
        self.vhl_spls.showCounts(func_print=print)
        func_print("* IS Sample Counts")
        self.is_spls.showCounts(func_print=print)
        func_print("* WS Sample Counts")
        self.ws_spls.showCounts(func_print=print)
        func_print("* Test")
        func_print(_sampleTestCounts(*self.spls_test))

    def toDict(self):
        to_dict = {
            "keys": self.keys,
            "data_scale": self.data_scale.toDict(),

            "vhl_spls": self.vhl_spls.toDict(),
            "is_spls": self.is_spls.toDict(),
            "ws_spls": self.ws_spls.toDict(),

            "vhl_fd": self.vhl_fd.toDict(),
            "is_fd": self.is_fd.toDict(),
            "ws_fd": self.ws_fd.toDict(),

            "test": [spl.toDict() for spl in self.spls_test]
        }

        return to_dict

    def loadDict(self, to_dict, *args, **kwargs):
        self.keys = to_dict["keys"]
        self.data_scale = _DataScale().loadDict(to_dict["data_scale"])

        self.vhl_spls = _MLSamples().loadDict(to_dict["vhl_spls"])
        self.is_spls = _MLSamples().loadDict(to_dict["is_spls"])
        self.ws_spls = _MLSamples().loadDict(to_dict["ws_spls"])

        self.vhl_fd = _FieldData().loadDict(to_dict["vhl_fd"])
        self.is_fd = _FieldData().loadDict(to_dict["is_fd"])
        self.ws_fd = _FieldData().loadDict(to_dict["ws_fd"])

        self.spls_test = [_Sample().loadDict(spl_dict) for spl_dict in to_dict["test"]]
        return self


class SamplesData(_SamplesData):

    def __init__(self):
        super(SamplesData, self).__init__()

    def mlfc(
            self, data_scale=DataScale(),
            vhl_x_keys=None, vhl_train_filters=None, vhl_test_filter=None, vhl_map_dict=None,
            is_x_keys=None, is_train_filters=None, is_test_filter=None, is_map_dict=None,
            ws_x_keys=None, ws_train_filters=None, ws_test_filter=None, ws_map_dict=None,
            test_filter=None, test_map_dict=None,
    ):
        if test_filter is None:
            test_filter = [("TEST", "==", "0")]
        if test_map_dict is None:
            test_map_dict = {"IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4}

        def _init(_x_keys, _train_filters, _test_filter, _map_dict,
                  _x_keys_tmp, _train_filters_tmp, _test_filter_tmp, _map_dict_tmp, ):
            def _is_none(_k, _k_tmp):
                if _k is None:
                    return _k_tmp
                return _k

            _x_keys = _is_none(_x_keys, _x_keys_tmp)
            _train_filters = _is_none(_train_filters, _train_filters_tmp)
            _test_filter = _is_none(_test_filter, _test_filter_tmp)
            _map_dict = _is_none(_map_dict, _map_dict_tmp)

            return _x_keys, _train_filters, _test_filter, _map_dict

        vhl_x_keys, vhl_train_filters, vhl_test_filter, vhl_map_dict = _init(
            vhl_x_keys, vhl_train_filters, vhl_test_filter, vhl_map_dict,
            vhl_x_keys, [], [], {
                "IS": 1, "VEG": 2, "SOIL": 1, "WAT": 3,
                "IS_SH": 3, "VEG_SH": 3, "SOIL_SH": 3, "WAT_SH": 3
            }
        )

        is_x_keys, is_train_filters, is_test_filter, is_map_dict = _init(
            is_x_keys, is_train_filters, is_test_filter, is_map_dict,
            is_x_keys, [], [], {"IS": 1, "SOIL": 2, }
        )

        ws_x_keys, ws_train_filters, ws_test_filter, ws_map_dict = _init(
            ws_x_keys, ws_train_filters, ws_test_filter, ws_map_dict,
            ws_x_keys, [], [], {"WAT": 3, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 2}
        )

        train_spls = _funcFilter([("TEST", "==", 1)], self.samples)
        test_spls = _funcFilter([("TEST", "==", 0)], self.samples)

        x_keys = list({*vhl_x_keys, *is_x_keys, *ws_x_keys})

        def _ml_spl(_x_keys, _train_filters, _test_filter, _map_dict):
            ml_spl = _MLSamples()

            def _func_filter(_filter, _samples):
                spls = _funcFilter(_filter, _samples, _map_dict)
                x = [spl.gets(x_keys) for spl in spls]
                y = [spl.code(_map_dict) for spl in spls]
                return x, y, spls

            ml_spl.x_train, ml_spl.y_train, ml_spl.spls_train = _func_filter(_train_filters, train_spls)
            ml_spl.x_test, ml_spl.y_test, ml_spl.spls_test = _func_filter(_test_filter, test_spls)
            ml_spl.keys = _x_keys
            ml_spl.data_scale = data_scale
            ml_spl.deal()

            return ml_spl

        mlfc_spl = _MLFCSamples()
        mlfc_spl.keys = x_keys
        mlfc_spl.data_scale = data_scale
        mlfc_spl.spls_test = _funcFilter(test_filter, self.samples, test_map_dict)
        mlfc_spl.x_test = [spl.gets(x_keys) for spl in mlfc_spl.spls_test]
        mlfc_spl.y_test = [spl.code(test_map_dict) for spl in mlfc_spl.spls_test]

        mlfc_spl.vhl_spls = _ml_spl(vhl_x_keys, vhl_train_filters, vhl_test_filter, vhl_map_dict)
        mlfc_spl.is_spls = _ml_spl(is_x_keys, is_train_filters, is_test_filter, is_map_dict)
        mlfc_spl.ws_spls = _ml_spl(ws_x_keys, ws_train_filters, ws_test_filter, ws_map_dict)
        mlfc_spl.vhl_fd = _FieldData(x_keys).initGetName(vhl_x_keys)
        mlfc_spl.is_fd = _FieldData(x_keys).initGetName(is_x_keys)
        mlfc_spl.ws_fd = _FieldData(x_keys).initGetName(ws_x_keys)

        mlfc_spl.deal()
        return mlfc_spl


class FCModel(_ModelInit):

    def __init__(self):
        super(FCModel, self).__init__()
        self.name = "FCModel"

        self.vhl_fd = _FieldData()
        self.is_fd = _FieldData()
        self.ws_fd = _FieldData()

        self.vhl_ml_mod = _ModelInit()
        self.intiVHLModel()

        self.is_ml_mod = _ModelInit()
        self.initISModel()

        self.ws_ml_mod = _ModelInit()
        self.initWSModel()

        self.test_filters: list[tuple[str, str, Union[int, str, float]]] = [("TEST", "==", 0)]

    def showCM(self):
        print("* VHL CM")
        print(self.vhl_ml_mod.cm().fmtCM())
        print("* IS CM")
        print(self.is_ml_mod.cm().fmtCM())
        print("* WS CM")
        print(self.ws_ml_mod.cm().fmtCM())

    def initWSModel(self, *args, **kwargs):
        return self

    def initISModel(self, *args, **kwargs):
        return self

    def intiVHLModel(self, *args, **kwargs):
        return self


class MLFCModel(FCModel):

    def __init__(self, name="MLFCModel"):
        super().__init__()
        self.name = "MLFCModel"

        self.name = name
        self.map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        self.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255)}
        self.cm_names = ["IS", "VEG", "SOIL", "WAT"]

        self.samples = _MLFCSamples()
        self.vhl_ml_mod = MLModel()
        self.intiVHLModel()
        self.is_ml_mod = MLModel()
        self.initISModel()
        self.ws_ml_mod = MLModel()
        self.initWSModel()

    def initWSModel(self, *args, **kwargs):
        self.ws_ml_mod.name = "WSMod"
        self.ws_ml_mod.filename = None
        self.ws_ml_mod.x_keys = [*SHH2Config.FEAT_NAMES.ALL]
        self.ws_ml_mod.map_dict = {"WAT": 3, "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 2}
        self.ws_ml_mod.data_scale = self.data_scale
        self.ws_ml_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), }
        self.ws_ml_mod.clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
        self.ws_ml_mod.train_filters = [("WS", "==", 1)]
        self.ws_ml_mod.test_filters = [("WS", "==", 1)]
        self.ws_ml_mod.cm_names = ["IS_SH", "NOIS_SH", "WAT"]
        return self

    def initISModel(self, *args, **kwargs):
        self.is_ml_mod.name = "ISMod"
        self.is_ml_mod.filename = None
        self.is_ml_mod.x_keys = FEAT_NAMES.OPT + FEAT_NAMES.OPT_GLCM + FEAT_NAMES.AS_BS + FEAT_NAMES.DE_BS
        self.is_ml_mod.map_dict = {"IS": 1, "SOIL": 2, }
        self.is_ml_mod.data_scale = self.data_scale
        self.is_ml_mod.color_table = {1: (255, 0, 0), 2: (255, 255, 0)}
        self.is_ml_mod.clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
        self.is_ml_mod.train_filters = [("IS", "==", 1)]
        self.is_ml_mod.test_filters = [("IS", "==", 1)]
        self.is_ml_mod.cm_names = ["IS", "SOIL"]
        return self

    def intiVHLModel(self, *args, **kwargs):
        self.vhl_ml_mod.name = "VHLMod"
        self.vhl_ml_mod.filename = None
        self.vhl_ml_mod.x_keys = [*FEAT_NAMES.OPT]
        self.vhl_ml_mod.map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 1, "WAT": 3,
            "IS_SH": 3, "VEG_SH": 3, "SOIL_SH": 3, "WAT_SH": 3
        }
        self.vhl_ml_mod.data_scale = self.data_scale
        self.vhl_ml_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 0), }
        self.vhl_ml_mod.clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
        self.vhl_ml_mod.train_filters = [("VHL", "==", 1)]
        self.vhl_ml_mod.test_filters = [("VHL", "==", 1)]
        self.vhl_ml_mod.cm_names = ["HIGH", "VEG", "LOW"]
        return self

    def sampleData(self, sd):
        self.samples = sd.mlfc(
            data_scale=self.data_scale,
            vhl_x_keys=self.vhl_ml_mod.x_keys, vhl_train_filters=self.vhl_ml_mod.train_filters,
            vhl_test_filter=self.vhl_ml_mod.test_filters, vhl_map_dict=self.vhl_ml_mod.map_dict,
            is_x_keys=self.is_ml_mod.x_keys, is_train_filters=self.is_ml_mod.train_filters,
            is_test_filter=self.is_ml_mod.test_filters, is_map_dict=self.is_ml_mod.map_dict,
            ws_x_keys=self.ws_ml_mod.x_keys, ws_train_filters=self.ws_ml_mod.train_filters,
            ws_test_filter=self.ws_ml_mod.test_filters, ws_map_dict=self.ws_ml_mod.map_dict,
            test_filter=self.test_filters, test_map_dict=self.map_dict
        )

        self.vhl_fd = self.samples.vhl_fd
        self.is_fd = self.samples.is_fd
        self.ws_fd = self.samples.ws_fd
        self.vhl_ml_mod.samples = self.samples.vhl_spls
        self.is_ml_mod.samples = self.samples.is_spls
        self.ws_ml_mod.samples = self.samples.ws_spls
        self.x_keys = self.samples.keys
        return self

    def train(self, *args, **kwargs):
        self.vhl_ml_mod.train()
        self.is_ml_mod.train()
        self.ws_ml_mod.train()
        self.accuracy(self.samples.x_test, self.samples.y_test)

    def predict(self, x, *args, **kwargs):
        self.vhl_fd.dim = 1
        self.is_fd.dim = 1
        self.ws_fd.dim = 1
        y = np.zeros(len(x))
        x_vhl = self.vhl_fd.get(x)
        y_vhl = self.vhl_ml_mod.predict(x_vhl)  # 1:IS|SOIL 2:VEG 3
        y[y_vhl == 2] = 2
        is_select = y_vhl == 1
        if np.sum(is_select * 1) != 0:
            x_is = x[is_select]
            x_is = self.is_fd.get(x_is)
            y_is = self.is_ml_mod.predict(x_is)
            y_is[y_is == 2] = 3
            y[is_select] = y_is
        ws_select = y_vhl == 3
        if np.sum(ws_select * 1) != 0:
            x_ws = x[ws_select]
            x_ws = self.ws_fd.get(x_ws)
            y_ws = self.ws_ml_mod.predict(x_ws)
            y_ws[y_ws == 3] = 4
            y[ws_select] = y_ws
        return y

    def toDict(self):
        to_dict = super(MLFCModel, self).toDict()
        to_dict = {
            **to_dict,
            "samples": self.samples.toDict(),

            "vhl_ml_mod": self.vhl_ml_mod.toDict(),
            "is_ml_mod": self.is_ml_mod.toDict(),
            "ws_ml_mod": self.ws_ml_mod.toDict(),

            "vhl_fd": self.vhl_fd.toDict(),
            "is_fd": self.is_fd.toDict(),
            "ws_fd": self.ws_fd.toDict(),

            "test_filter": self.test_filters,
        }
        return to_dict

    def loadDict(self, to_dict):
        super(MLFCModel, self).loadDict(to_dict)
        self.samples = _MLFCSamples().loadDict(to_dict["samples"])

        self.vhl_ml_mod.loadDict(to_dict["vhl_ml_mod"])
        self.is_ml_mod.loadDict(to_dict["is_ml_mod"])
        self.ws_ml_mod.loadDict(to_dict["ws_ml_mod"])

        self.vhl_fd.loadDict(to_dict["vhl_fd"])
        self.is_fd.loadDict(to_dict["is_fd"])
        self.ws_fd.loadDict(to_dict["ws_fd"])

        self.test_filters = to_dict["test_filter"]

    def save(self, filename=None, dirname=None, is_save_clf=True, is_save_data=True, *args, **kwargs):
        filename = self._getfilename(dirname, filename)

        if is_save_clf or is_save_clf:
            to_dirname = self._gettodirname(filename)
            self.vhl_ml_mod.save(dirname=to_dirname, is_save_clf=is_save_clf, is_save_data=is_save_data)
            self.is_ml_mod.save(dirname=to_dirname, is_save_clf=is_save_clf, is_save_data=is_save_data)
            self.ws_ml_mod.save(dirname=to_dirname, is_save_clf=is_save_clf, is_save_data=is_save_data)

        if is_save_data:
            to_dirname = self._gettodirname(filename)
            to_data_fn = os.path.join(to_dirname, "{}_data.npy")
            np.save(to_data_fn.format("x_test"), np.array(self.samples.x_test))
            np.save(to_data_fn.format("y_test"), np.array(self.samples.y_test))

        to_dict = self.toDict()
        to_dict["__class__.__name__"] = self.__class__.__name__
        saveJson(to_dict, filename)

    def load(self, filename=None, dirname=None, *args, **kwargs):
        filename = self._getfilename(dirname, filename)
        to_dict = readJson(filename)
        self.loadDict(to_dict)

        to_dirname = self._gettodirname(filename, False)
        self.vhl_ml_mod.load(dirname=to_dirname)
        self.is_ml_mod.load(dirname=to_dirname)
        self.ws_ml_mod.load(dirname=to_dirname)

        to_data_fn = os.path.join(to_dirname, "{}_data.npy")

        def loaddata(_fn):
            if os.path.isfile(_fn):
                return np.load(_fn)
            else:
                return None

        self.samples.x_test = loaddata(to_data_fn.format("x_test"))
        self.samples.y_test = loaddata(to_data_fn.format("y_test"))
        return self

    def imdc(self, raster_fns, to_imdc_fn=None, *args, **kwargs):
        to_imdc_fn = _imdc1(self.filename, self, self.x_keys, raster_fns,
                            self.data_scale, self.color_table, to_imdc_fn)
        return to_imdc_fn


def main():
    def func1():
        cm_names = ["IS", "VEG", "SOIL", "WAT"]

        sd = SamplesData()
        sd.addCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd5_spl.csv")

        ml_mod = MLModel()
        ml_mod.filename = r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp9.shh2mod"
        ml_mod.x_keys = SHH2Config.NAMES
        ml_mod.map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        ml_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        ml_mod.clf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=1, min_samples_split=2)
        ml_mod.test_filters = [("TAG", "==", "shh2_spl26_4_random800_spl2")]
        ml_mod.sampleData(sd)
        ml_mod.samples.showCounts()
        ml_mod.train()
        ml_mod.score(ml_mod.samples.x_test, ml_mod.samples.y_test)
        print("cm", ml_mod.cm(cm_names).accuracyCategory("IS").fmtCM(), sep="\n")
        ml_mod.save()
        ml_mod.imdc(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions\QDTR0.tif")

        ml_mod = MLModel().load(r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp9.shh2mod")
        ml_mod.imdc(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions\QDTR1.tif")

    def func2():
        cm_names = ["IS", "VEG", "SOIL", "WAT"]
        sd = SamplesData()
        sd.addCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv")
        mlfc_mod = MLFCModel()
        mlfc_mod.filename = r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp10.shh2mod"
        mlfc_mod.test_filters.append(("TAG", "==", "shh2_spl26_4_random800_spl2"))
        mlfc_mod.sampleData(sd)
        mlfc_mod.samples.showCounts()
        mlfc_mod.train()
        mlfc_mod.showCM()
        print("cm", mlfc_mod.cm(cm_names).accuracyCategory("IS").fmtCM(), sep="\n")
        mlfc_mod.save()

        mlfc_mod = MLFCModel().load(r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp10.shh2mod")
        mlfc_mod.samples.showCounts()
        mlfc_mod.imdc(SHH2Config.CD_ENVI_FN)

    def func3():
        cm_names = ["IS", "VEG", "SOIL", "WAT"]
        get_names = [
            "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
            "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
            "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
        ]
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv"
        sd = SamplesData()
        sd.addDLCSV(csv_fn, (21, 21), get_names)

        torch_mod = TorchModel()
        torch_mod.filename = None
        torch_mod.map_dict = {
            "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
            "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
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
        torch_mod.epochs = 100
        torch_mod.train_filters.append(("city", "==", "cd"))
        torch_mod.test_filters.append(("city", "==", "cd"))
        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts()
        torch_mod.save_model_fmt = r"F:\Week\20240707\Data\model2\model{}.pth"
        torch_mod.train()

        mod_fn = None
        if mod_fn is not None:
            torch_mod.imdc([
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_1_4.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_2_4.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_1.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_2.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_3.tif",
                r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles\SHH2_CD2_envi_3_4.tif",
            ], mod_fn=mod_fn)

    def func4():
        cm_names = ["IS", "VEG", "SOIL", "WAT"]

        sd = SamplesData()
        sd.addCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv")

        ml_mod = MLModel()
        ml_mod.filename = r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp14.shh2mod"
        ml_mod.x_keys = ["Red", "Blue", "Green", "NIR"]
        ml_mod.map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        ml_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        ml_mod.data_scale.readJson(SHH2Config.CD_RANGE_FN)
        ml_mod.clf = SVC(kernel="rbf", C=8.42, gamma=0.127)
        ml_mod.test_filters = [("TAG", "==", "shh2_spl26_4_random800_spl2")]
        ml_mod.sampleData(sd)
        ml_mod.samples.showCounts()
        ml_mod.train()
        ml_mod.score(ml_mod.samples.x_test, ml_mod.samples.y_test)
        print("cm", ml_mod.cm(cm_names).accuracyCategory("IS").fmtCM(), sep="\n")
        # ml_mod.save()
        ml_mod.imdc(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions\QDTR1.tif")

        # ml_mod.imdc(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions\QDTR0.tif")
        # ml_mod = MLModel().load(r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp9.shh2mod")
        # ml_mod.imdc(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\TestRegions\QDTR1.tif")

    func3()
    return


if __name__ == "__main__":
    main()
