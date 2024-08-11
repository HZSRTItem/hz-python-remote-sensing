# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTModel.py
@Time    : 2023/12/9 21:53
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTModel
-----------------------------------------------------------------------------"""
import os
import os.path
import random
import warnings
from datetime import datetime
from typing import Union

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from torch import optim, nn
from torch.utils.data import Dataset

from SRTCodes.GDALTorch import GDALTorchImdc
from SRTCodes.GDALUtils import samplingCSVData
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.NumpyUtils import NumpyDataCenter
from SRTCodes.PytorchModelTraining import TorchTraining
from SRTCodes.SRTModelImage import GDALImdc
from SRTCodes.Utils import datasCaiFen, changext, Jdt, saveJson, readJson, DirFileName, FN, printLinesStringTable

_DL_SAMPLE_DIRNAME = r"F:\Data\DL"



class _SHH2Config:

    def __init__(self):
        super(_SHH2Config, self).__init__()


class SRTModelInit:

    def __init__(self):
        self.name = "MODEL"
        self.save_dict = {}

    def train(self, *args, **kwargs):
        train_args = {"name": self.name}
        return train_args

    def predict(self, *args, **kwargs):
        return 0

    def score(self, *args, **kwargs):
        return 0

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True

    def __getitem__(self, item):
        return self.save_dict[item]

    def __setitem__(self, key, value):
        self.save_dict[key] = value

    def saveDict(self, filename, *args, **kwargs):
        for k in kwargs:
            self.save_dict[k] = kwargs[k]
        self.save_dict["__BACK__"] = list(args)
        saveJson(self.save_dict, filename)

    def loadDict(self, filename):
        self.save_dict = readJson(filename)


class SRTHierarchicalModel(SRTModelInit):

    def __init__(self):
        super(SRTHierarchicalModel, self).__init__()

        self.models = {}
        self.train_data = {}
        self.test_data = {}
        self.save_dirname = None
        self.formatted_time = None

    def saveDirName(self, dirname=None):
        self.save_dirname = dirname
        if not os.path.isdir(self.save_dirname):
            os.mkdir(self.save_dirname)

    def timeDirName(self, dirname=None):
        if dirname is not None:
            self.save_dirname = dirname
        current_time = datetime.now()
        self.formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        self.save_dirname = os.path.join(self.save_dirname, self.formatted_time)
        if not os.path.isdir(self.save_dirname):
            os.mkdir(self.save_dirname)

    def model(self, name, mod, *args, **kwargs):
        self.models[name] = mod

    def trainData(self, name, x, y, *args, **kwargs):
        self.train_data[name] = (x, y)

    def testData(self, name, x, y, *args, **kwargs):
        self.test_data[name] = (x, y)

    def train(self, *args, **kwargs):
        for name in self.models:
            x, y = self.train_data[name]
            self.models[name].train()

    def save(self, *args, **kwargs):
        for name in self.models:
            self.models[name].save(os.path.join(self.save_dirname, name + ".mod"))

    def load(self, dirname, *args, **kwargs):
        for name in self.models:
            self.models[name].load(os.path.join(dirname, name + ".mod"))


def mapDict(data, map_dict, return_select=False):
    if map_dict is None:
        return data
    to_list = []
    select_list = []
    for i, d in enumerate(data):
        if d in map_dict:
            to_list.append(map_dict[d])
        select_list.append(d in map_dict)
    if return_select:
        return to_list, select_list
    return to_list


class _RandomGridSearch:

    def __init__(self, n_run, **kwargs):
        self.n_run = n_run
        self.kwargs = kwargs
        self.testing_list = []

        self.bast_kwargs = None
        self.bast_accuracy = 0

    def __getitem__(self, item):
        return self.kwargs[item]

    def __setitem__(self, key, value):
        self.kwargs[key] = value

    def __len__(self):
        return len(self.kwargs)

    def __contains__(self, item):
        return item in self.kwargs

    def ml(self, model_cls, x, y, scoring=None, n_cv=10, is_return_bast_clf=True, ):
        self.bast_accuracy = 0.0
        if self._number() < self.n_run:
            self.n_run = self._number()
        jdt = Jdt(self.n_run, "Random Grid Search").start()
        for n in range(self.n_run):
            kwargs = self.generate_params()
            if kwargs is None:
                warnings.warn("All run {} > {}".format(self._number(), self.n_run))
            clf = model_cls(**kwargs)
            scores = cross_val_score(estimator=clf, X=x, y=y, cv=n_cv, scoring=scoring, n_jobs=-1)
            if scores.mean() > self.bast_accuracy:
                self.bast_kwargs = kwargs
                self.bast_accuracy = scores.mean()
            jdt.add()
        jdt.end()
        if is_return_bast_clf:
            clf = model_cls(**self.bast_kwargs)
            clf.fit(x, y)
            return clf
        return None

    def _number(self):
        n = 1
        for k in self.kwargs:
            n *= len(self.kwargs[k])
        return n

    def generate_params(self):

        def _generate():
            _to_dict = {}
            for k in self.kwargs:
                _to_dict[k] = random.choice(self.kwargs[k])
            return _to_dict

        n_all = self._number()
        for _ in range(n_all * 100):
            is_return = True
            to_dict = _generate()
            for to_dict_tmp in self.testing_list:
                if set(to_dict_tmp.items()) == set(to_dict.items()):
                    is_return = False
                    break
            if is_return:
                self.testing_list.append(to_dict)
                return to_dict
        return None


class _SK_MOD:

    def __init__(self):
        self.clf = None

    def predict(self, x, *args, **kwargs):
        return self.clf.predict(x)

    def score(self, x, y, *args, **kwargs):
        return self.clf.score(x, y)

    def fit(self, x, y, *args, **kwargs):
        return None


def _rgsML(model_cls, x, y, scoring=None, n_cv=10, n_run=20):
    rgs = _RandomGridSearch(n_run)
    if model_cls == SVC:
        rgs["C"] = [0.1, 1, 10, 100]
        rgs["gamma"] = [0.01, 0.1, 1, 10]
    elif model_cls == RandomForestClassifier:
        rgs["n_estimators"] = [50, 70, 90, 100, 110, 120, 150, 180]
        rgs["max_depth"] = [i for i in range(3, 10)]
        rgs["min_samples_split"] = [i for i in range(2, 10)]
        rgs["min_samples_leaf"] = [i for i in range(1, 6)]
    model = rgs.ml(model_cls, x, y, scoring=scoring, n_cv=n_cv, is_return_bast_clf=True)
    return model, rgs.bast_kwargs, rgs.bast_accuracy


class _RGS(_SK_MOD):

    def __init__(self):
        super().__init__()

        self.bast_model = None
        self.bast_kwargs = None
        self.bast_accuracy = 0
        self.n_cv = 10
        self.n_run = 20
        self.model_cls = None

    def fit(self, x, y, *args, **kwargs):
        self.bast_model, self.bast_kwargs, self.bast_accuracy = _rgsML(
            self.model_cls, x, y, n_cv=self.n_cv, n_run=self.n_run)
        self.clf = self.bast_model


class RF_RGS(_RGS):

    def __init__(self):
        super().__init__()
        self.model_cls = RandomForestClassifier


class SVM_RGS(_RGS):

    def __init__(self):
        super().__init__()
        self.model_cls = SVC


class _FieldData:

    def __init__(self, *names, dim=0):
        self.names = list(datasCaiFen(names))
        self.get_names = []
        self.dim = dim
        self.find_list = []

    def addName(self, *names):
        self.names.extend(datasCaiFen(names))
        return self

    def initGetName(self, *names):
        self.get_names = list(datasCaiFen(names))
        self.find_list = [self.names.index(name) for name in self.get_names]
        return self

    def get(self, data):
        if self.dim == 0:
            return data[self.find_list]
        if self.dim == 1:
            return data[:, self.find_list]
        if self.dim == 2:
            return data[:, :, self.find_list]
        return None

    def toDict(self):
        return {
            "names": self.names,
            "get_names": self.get_names,
            "dim": self.dim,
            "find_list": self.find_list,
        }

    def loadDict(self, to_dict):
        self.names = to_dict["names"]
        self.get_names = to_dict["get_names"]
        self.dim = to_dict["dim"]
        self.find_list = to_dict["find_list"]
        return self


class DataScale:

    def __init__(self, is01=True):
        self.datas = {}
        self.is01 = is01

    def add(self, name, d_min=None, d_max=None, is01=None):
        if is01 is None:
            is01 = self.is01
        self.datas[name] = {"min": d_min, "max": d_max, "is01": is01}

    def fits(self, data, find_list=None):
        if find_list is None:
            find_list = [k for k in data]
        for k in find_list:
            if k in self.datas:
                data[k] = self.fit(k, data[k])
        return data

    def fit(self, name, data):
        if name not in self.datas:
            return data

        x_min, x_max = self.datas[name]["min"], self.datas[name]["max"]
        if x_min is None:
            x_min = np.min(data)
        if x_max is None:
            x_max = np.max(data)

        data = np.clip(data, x_min, x_max, )
        if self.datas[name]["is01"]:
            data = (data - x_min) / (x_max - x_min)
        return data

    def readJson(self, json_fn):
        json_dict = readJson(json_fn)
        for k in json_dict:
            self.add(k, json_dict[k]["min"], json_dict[k]["max"], )
        return self

    def toDict(self):
        return {"datas": self.datas, "is01": self.is01, }

    def loadDict(self, to_dict):
        self.datas = to_dict["datas"]
        self.is01 = to_dict["is01"]
        return self


class FeatFuncs:

    def __init__(self):
        self.funcs = {}

    def add(self, name, func):
        if name not in self.funcs:
            self.funcs[name] = []
        self.funcs[name].append(func)
        return self

    def fits(self, data):
        for name in data:
            data[name] = self.fit(name, data[name])
        return data

    def fit(self, name, data):
        if name in self.funcs:
            funcs = self.funcs[name]
            for func in funcs:
                data = func(data)
        return data

    def __contains__(self, item):
        return item in self.funcs

    def __len__(self):
        return len(self.funcs)

    def __str__(self):
        to_str = "FeatFuncs("
        for i, k in enumerate(self.funcs):
            to_str += "\n{:>2}. \"{}\": {}".format(i + 1, k, self.funcs[k])
        to_str += ")"
        return to_str


class _Sample:

    def __init__(self, cname=None, uid=None, x=None, y=None, field_datas=None, data=None,
                 data_keys=None, data_get_keys=None, x_deal=None):
        if field_datas is None:
            field_datas = {}
        self.x = x
        self.y = y
        self.uid = uid
        self.cname = cname
        self.data = data
        self.data_keys = data_keys
        self.field_datas = field_datas
        self.fd = _FieldData()

        self.initData(data, data_keys, data_get_keys=data_get_keys, x_deal=x_deal)
        self.data_deal = None

    def initData(self, data, data_keys, data_get_keys=None, x_deal=None):
        if data is None:
            return self
        if data_keys is None:
            data_keys = [i for i in range(len(data))]
        self.data = data
        self.data_keys = data_keys
        self.fd = _FieldData(*data_keys)
        if data_get_keys is not None:
            self.fd.initGetName(*data_get_keys)
        self.data_deal = x_deal

    def initDict(self, line, data=None, data_keys=None, data_get_keys=None, x_deal=None):
        if "X" in line:
            self.x = float(line["X"])
        if "Y" in line:
            self.y = float(line["Y"])
        if "SRT" in line:
            self.uid = int(line["SRT"])
        if "CNAME" in line:
            self.cname = str(line["CNAME"])

        self.field_datas = line.to_dict()
        self.initData(data, data_keys, data_get_keys=data_get_keys, x_deal=x_deal)
        return self

    def __getitem__(self, item):
        if "X" == item:
            return self.x
        if "Y" == item:
            return self.y
        if "SRT" == item:
            return self.uid
        if "CNAME" == item:
            return self.cname
        return self.field_datas[item]

    def filter(self, filter_data):
        fn, ft, d = filter_data
        if ft == ">":
            return self.__getitem__(fn) > d
        if ft == ">=":
            return self.__getitem__(fn) >= d
        if ft == "<":
            return self.__getitem__(fn) < d
        if ft == "<=":
            return self.__getitem__(fn) <= d
        if ft == "==":
            return self.__getitem__(fn) == d
        if ft == "!=":
            return self.__getitem__(fn) != d
        return False

    def filters(self, *filter_datas):
        for filter_data in filter_datas:
            if not self.filter(filter_data):
                return False
        return True

    def __contains__(self, item):
        if "X" == item:
            return True
        if "Y" == item:
            return True
        if "SRT" == item:
            return True
        if "CNAME" == item:
            return True
        return item in self.field_datas

    def gets(self, *keys):
        keys = datasCaiFen(keys)
        to_dict = {}
        for k in keys:
            if self.__contains__(k):
                to_dict[k] = self.__getitem__(k)
            else:
                to_dict[k] = None
        return to_dict

    def code(self, map_dict):
        return map_dict[self.cname]

    def toDict(self):
        line = self.field_datas
        line["X"] = self.x
        line["Y"] = self.y
        line["SRT"] = self.uid
        line["CNAME"] = self.cname
        return line

    def loadDict(self, to_dict):
        self.x = to_dict["X"]
        self.y = to_dict["Y"]
        self.uid = to_dict["SRT"]
        self.cname = to_dict["CNAME"]
        self.field_datas = to_dict
        return self

    def getdata(self):
        x = self.fd.get(self.data)
        if self.data_deal is not None:
            x = self.data_deal(x)
        return x


def _funcFilter(_filter, samples, map_dict=None):
    spls = []
    for spl in samples:
        spl: _Sample
        if map_dict is not None:
            if spl.cname not in map_dict:
                continue
        if not spl.filters(*_filter):
            continue
        spls.append(spl)
    return spls


class _Samples:

    def __init__(self):
        self.keys = []
        self.data_scale = DataScale()
        self.feat_funcs = FeatFuncs()
        self.fd = _FieldData()

        self.x_train = None
        self.y_train = None
        self.spls_train = None

        self.x_test = None
        self.y_test = None
        self.spls_test = None

    def deal(self):
        return self

    def toDict(self):
        return {
            "keys": self.keys,
            "data_scale": self.data_scale.toDict(),
        }

    def loadDict(self, to_dict, *args, **kwargs):
        self.keys = to_dict["keys"]
        self.data_scale = DataScale().loadDict(to_dict["data_scale"])

    def __len__(self):
        return len(self.keys)

    def showCounts(self, func_print=print, *args, **kwargs):
        return None


def _sampleTestCounts(*samples):
    def samplesDescription(df):
        df_des = pd.DataFrame({
            "Training": df[df["TEST"] == 1].groupby("CNAME").count()["TEST"].to_dict(),
            "Testing": df[df["TEST"] == 0].groupby("CNAME").count()["TEST"].to_dict()
        })
        df_des[pd.isna(df_des)] = 0
        df_des["SUM"] = df_des.apply(lambda x: x.sum(), axis=1)
        df_des.loc["SUM"] = df_des.apply(lambda x: x.sum())
        return df_des

    samples = datasCaiFen(samples)
    df = pd.DataFrame([{"CNAME": spl.cname, "TEST": spl["TEST"]} for spl in samples])
    df = samplesDescription(df)
    return df


class _MLSamples(_Samples):

    def __init__(self):
        super().__init__()

    def deal(self):
        df_train = pd.DataFrame(self.x_train)
        df_train = self.feat_funcs.fits(df_train)
        self.x_train = self.data_scale.fits(df_train)
        self.x_train = self.x_train.values
        df_test = pd.DataFrame(self.x_test)
        df_test = self.feat_funcs.fits(df_test)
        self.x_test = self.data_scale.fits(df_test)
        self.x_test = self.x_test.values
        return self

    def dataDescription(self, fmt="{:>.3f}", print_func=print, end=""):
        lines = [["NAME", "MIN 1", "MAX 1", "MEAN 1", "STD 1", "MIN 1", "MAX 0", "MEAN 0", "STD 0", ]]
        for i, k in enumerate(self.keys):
            lines.append([
                k,
                fmt.format(np.min(self.x_train[:, i])),
                fmt.format(np.max(self.x_train[:, i])),
                fmt.format(np.mean(self.x_train[:, i])),
                fmt.format(np.std(self.x_train[:, i])),
                fmt.format(np.min(self.x_test[:, i])),
                fmt.format(np.max(self.x_test[:, i])),
                fmt.format(np.mean(self.x_test[:, i])),
                fmt.format(np.std(self.x_test[:, i])),
            ])
        printLinesStringTable(lines, ">", print_func=print_func)
        print_func(end=end)

    def data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test

    def traindata(self):
        return self.x_train, self.y_train

    def testdata(self):
        return self.x_test, self.y_test

    def toDict(self):
        return {
            "keys": self.keys,
            "data_scale": self.data_scale.toDict(),
            "train": [spl.toDict() for spl in self.spls_train],
            "test": [spl.toDict() for spl in self.spls_test],
        }

    def loadDict(self, to_dict, *args, **kwargs):
        self.keys = to_dict["keys"]
        self.spls_train = [_Sample().loadDict(spl_dict) for spl_dict in to_dict["train"]]
        self.spls_test = [_Sample().loadDict(spl_dict) for spl_dict in to_dict["test"]]
        self.data_scale = DataScale().loadDict(to_dict["data_scale"])
        return self

    def showCounts(self, func_print=print, *args, **kwargs):
        df = _sampleTestCounts(*self.spls_train, *self.spls_test)
        if "is_show" in kwargs:
            if not kwargs["is_show"]:
                return df
        func_print(df)
        return df


class TorchDataset(Dataset):

    def __init__(self, n_datas=None, n_channels=None, win_size=None, read_size=None, device="cuda"):
        super(TorchDataset, self).__init__()
        if (n_datas is None) or (n_channels is None):
            return
        self.data = torch.zeros(n_datas, n_channels, read_size[0], read_size[1], dtype=torch.float32, device=device)
        self.y = None
        if (win_size is None) and (read_size is None):
            return
        if win_size == read_size:
            self.ndc = None
        else:
            self.ndc = NumpyDataCenter(3, win_size, read_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        if self.ndc is not None:
            x = self.ndc.fit(x)
        y = self.y[item]
        return x, y


class _TorchSamples(_Samples):

    def __init__(self):
        super().__init__()

        self.train_ds = TorchDataset()
        self.test_ds = TorchDataset()

    def showCounts(self, func_print=print, *args, **kwargs):
        func_print(_sampleTestCounts(*self.spls_train, *self.spls_test))

    def deal(self):
        if len(self.spls_train) > 0:
            self.keys = self.fd.get_names


class SamplesData:

    def __init__(self, _dl_sample_dirname=_DL_SAMPLE_DIRNAME):
        self.samples = []
        self._dl_sample_dirname = _dl_sample_dirname

    def addDF(self, df, data=None, data_keys=None, data_get_keys=None, x_deal=None):
        for i in range(len(df)):
            if data is not None:
                self.samples.append(_Sample().initDict(
                    df.loc[i], data=data[i],
                    data_keys=data_keys, data_get_keys=data_get_keys, x_deal=x_deal
                ))
            else:
                self.samples.append(_Sample().initDict(
                    df.loc[i], data=data,
                    data_keys=data_keys, data_get_keys=data_get_keys, x_deal=x_deal
                ))
        return self

    def addCSV(self, csv_fn, data=None, data_keys=None):
        self.addDF(pd.read_csv(csv_fn), data=data, data_keys=data_keys)
        return self

    def addDLCSV(self, csv_fn, read_size, data_get_keys, x_deal=None, grs=None, ):
        dfn = DirFileName(self._dl_sample_dirname)
        csv_fn_spl = dfn.fn("{}-{}_{}.csv".format(FN(csv_fn).getfilenamewithoutext(), read_size[0], read_size[1]))
        npy_fn = FN(csv_fn_spl).changext("-data.npy")
        names_fn = FN(csv_fn_spl).changext("-names.json")

        if not os.path.isfile(csv_fn_spl):
            # csv_fn, to_fn, to_npy_fn, names_fn, win_rows, win_columns, gr
            samplingCSVData(csv_fn, csv_fn_spl, npy_fn, names_fn, read_size[0], read_size[1], grs, )

        df = pd.read_csv(csv_fn_spl)
        data = np.load(npy_fn)
        data_keys = readJson(names_fn)

        self.addDF(df, data=data, data_keys=data_keys, data_get_keys=data_get_keys, x_deal=x_deal)

    def ml(
            self, x_keys, map_dict, data_scale=DataScale(),
            train_filters=None, test_filter=None, feat_funcs=FeatFuncs(),
    ):
        train_filters, test_filter = self.gettraintestfilter(train_filters, test_filter)

        def _func_filter(_filter):
            x, y, spls = [], [], []
            for spl in self.samples:
                spl: _Sample
                if spl.cname not in map_dict:
                    continue
                if not spl.filters(*_filter):
                    continue
                x.append(spl.gets(x_keys))
                y.append(spl.code(map_dict))
                spls.append(spl)
            return x, y, spls

        ml_spl = _MLSamples()
        ml_spl.x_train, ml_spl.y_train, ml_spl.spls_train = _func_filter(train_filters)
        ml_spl.x_test, ml_spl.y_test, ml_spl.spls_test = _func_filter(test_filter)
        ml_spl.keys = x_keys
        ml_spl.data_scale = data_scale
        ml_spl.feat_funcs = feat_funcs
        ml_spl.deal()
        return ml_spl

    def gettraintestfilter(self, train_filters, test_filter, ):
        if train_filters is None:
            train_filters = [("TEST", "==", 1)]
        else:
            if ("TEST", "==", 1) not in train_filters:
                train_filters.append(("TEST", "==", 1))
        if test_filter is None:
            test_filter = [("TEST", "==", 0)]
        else:
            if ("TEST", "==", 0) not in test_filter:
                test_filter.append(("TEST", "==", 0))
        return train_filters, test_filter

    def dltorch(self, map_dict, win_size, read_size, train_filters=None, test_filter=None, device="cuda"):
        train_filters, test_filter = self.gettraintestfilter(train_filters, test_filter)
        train_spls = _funcFilter(train_filters, self.samples, map_dict=map_dict)
        test_spls = _funcFilter(test_filter, self.samples, map_dict=map_dict)

        torch_spl = _TorchSamples()
        torch_spl.fd = self.samples[0].fd

        def build_ds(_ds, _spls):
            _ds.__init__(len(_spls), len(torch_spl.fd.get_names), win_size, read_size, device=device)
            _y_list = []
            for i, _spl in enumerate(_spls):
                _spl: _Sample
                _ds.data[i] = torch.from_numpy(_spl.getdata().astype("float32")).to(device)
                _y_list.append(_spl.code(map_dict))
            _ds.y = torch.from_numpy(np.array(_y_list, dtype="int64")).to(device)

        torch_spl.spls_train = train_spls
        build_ds(torch_spl.train_ds, train_spls)
        torch_spl.spls_test = test_spls
        build_ds(torch_spl.test_ds, test_spls)

        torch_spl.deal()
        return torch_spl

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class _GDALImdc(GDALImdc):

    def __init__(self, *raster_fns, data_scale: DataScale = None, feat_funcs: FeatFuncs = None):
        super().__init__(*raster_fns)
        self.data_scale = data_scale
        self.feat_funcs = feat_funcs

    def readRaster(self, data, fit_names, gr, is_jdt, *args, **kwargs):
        jdt = Jdt(len(fit_names), "Read Raster").start(is_jdt)
        for i, name in enumerate(fit_names):
            data_i = gr.readGDALBand(name)
            data_i[np.isnan(data_i)] = 0
            if self.feat_funcs is not None:
                data_i = self.feat_funcs.fit(name, data_i)
            if self.data_scale is not None:
                data_i = self.data_scale.fit(name, data_i)
            data[i] = data_i
            jdt.add(is_jdt)
        jdt.end(is_jdt)


def _imdc1(fn, mod, x_keys, raster_fns, data_scale, color_table, to_imdc_fn, feat_funcs=None):
    if to_imdc_fn is None:
        to_imdc_fn = changext(fn, "_imdc.tif")
    gimdc = _GDALImdc(raster_fns, data_scale=data_scale, feat_funcs=feat_funcs)
    gimdc.imdc1(mod, to_imdc_fn, x_keys, color_table=color_table)
    return to_imdc_fn


def _imdc2(fn, mod, raster_fns, data_scale, win_size, to_imdc_fn,
           fit_names, color_table, n=1000, feat_funcs=None):
    if to_imdc_fn is None:
        to_imdc_fn = changext(fn, "_imdc.tif")
    gimdc = _GDALImdc(raster_fns, data_scale=data_scale, feat_funcs=feat_funcs)

    def func_predict(x):
        return mod.predict(x)
        # return np.zeros(len(x))

    gimdc.imdc2(func_predict=func_predict, win_size=win_size, to_imdc_fn=to_imdc_fn,
                fit_names=fit_names, data_deal=None, color_table=color_table, n=n)
    return to_imdc_fn


class _ModelInit:
    """
    Samples  -> training

    Model    -> training

    Accuracy -> training

    training:
        1. get samples
            ML:
                dataframe
                x keys

            DL: dataset, dataloader
        2. train

      ->func::predict(data)
      ->func::save(filename)
      ->func::load(filename)

      ML training:
        get:
            dataframe:
                x field name
                y field name
                category field name
                test field name
                srt field name
                map dict
    """

    def __init__(
            self,
            name="_ModelInit",
    ):
        self.samples = None
        self.name = name
        self.filename = None
        self.x_keys = []
        self.map_dict = {}
        self.data_scale = DataScale()
        self.feat_funcs = FeatFuncs()
        self.color_table = {}
        self.cm_names = None

        self.accuracy_dict = {}

        self.train_filters: list[tuple[str, str, Union[int, str, float]]] = []
        self.test_filters: list[tuple[str, str, Union[int, str, float]]] = []

        self.fd = _FieldData()

    def predict(self, *args, **kwargs):
        return np.array([])

    def train(self, *args, **kwargs):
        return

    def save(self, *args, **kwargs):
        return

    def load(self, *args, **kwargs):
        return

    def imdc(self, *args, **kwargs):
        raster_fn = None
        to_imdc_fn = None
        if len(args) == 1:
            raster_fn = args[0]
        if len(args) == 2:
            to_imdc_fn = args[1]
        if raster_fn is None:
            return None
        to_imdc_fn = _imdc1(
            self.filename, self, self.x_keys, raster_fn,
            self.data_scale, self.color_table, to_imdc_fn,
            feat_funcs=self.feat_funcs)
        return to_imdc_fn

    def score(self, x=None, y=None, *args, **kwargs):
        return accuracy_score(y, self.predict(x))

    def toDict(self, *args, **kwargs):
        to_dict = {
            "name": self.name,
            "filename": self.filename,
            "x_keys": self.x_keys,
            "map_dict": self.map_dict,
            "data_scale": self.data_scale.toDict(),
            "color_table": str(self.color_table),
            "accuracy_dict": self.accuracy_dict,
            "cm_names": self.cm_names,
        }
        return to_dict

    def _getfilename(self, dirname, filename):
        if filename is not None:
            filename = changext(filename, ".shh2mod")
        else:
            if dirname is not None:
                filename = os.path.join(dirname, "{}.shh2mod".format(self.name))
            else:
                filename = self.filename
        self.filename = filename
        return filename

    def _gettodirname(self, filename=None, is_mk=True):
        if filename is None:
            filename = self.filename
        to_dirname = os.path.splitext(filename)[0]
        if is_mk:
            if not os.path.isdir(to_dirname):
                os.mkdir(to_dirname)
        return to_dirname

    def loadDict(self, to_dict):
        self.name = to_dict["name"]
        self.x_keys = to_dict["x_keys"]
        self.map_dict = to_dict["map_dict"]
        self.data_scale = DataScale().loadDict(to_dict["data_scale"])
        self.color_table = eval(to_dict["color_table"])
        self.accuracy_dict = to_dict["accuracy_dict"]
        self.cm_names = to_dict["cm_names"]
        return self

    def cm(self, cm_names=None):
        if cm_names is None:
            cm_names = self.cm_names
        if cm_names is None:
            cm_names = list(set(self.map_dict.values()))
            cm_names.sort()
        cm = ConfusionMatrix(class_names=cm_names)
        cm.addData(self.accuracy_dict["y1"], self.accuracy_dict["y2"])
        return cm

    def csvSamples(self, csv_fn):
        sd = SamplesData()
        sd.addCSV(csv_fn)
        self.sampleData(sd)

    def sampleData(self, sd):
        return self

    def accuracy(self, x_test, y_test):
        y2 = self.predict(x_test)
        self.accuracy_dict = {
            "y1": y_test,
            "y2": y2.tolist(),
        }
        if self.samples is not None:
            self.accuracy_dict = {
                **self.accuracy_dict,
                "X": [spl.x for spl in self.samples.spls_test],
                "Y": [spl.y for spl in self.samples.spls_test],
                "SRT": [spl.uid for spl in self.samples.spls_test],
                "CNAME": [spl.cname for spl in self.samples.spls_test],
            }
        return self.accuracy_dict


class MLModel(_ModelInit):
    r"""
    Sklearn models. Contain training, image classification.

    Examples
    --------

    >>> # Samples
    >>> sd = SamplesData()
    >>> sd.addCSV("csv_fn")
    >>> # Model
    >>> ml_mod = MLModel()
    >>> ml_mod.name = "MLModel"
    >>> ml_mod.filename = r"*.shh2mod"
    >>> ml_mod.x_keys = ["FIELD_NAME", ...]
    >>> ml_mod.map_dict = {"CATEGORY_NAME": 1, ...}
    >>> ml_mod.data_scale = DataScale().readJson("json_fn")
    >>> ml_mod.color_table = {1: (255, 0, 0), ... }
    >>> ml_mod.clf = RandomForestClassifier()
    >>> ml_mod.train_filters = [("FIELD_NAME", "FT:[==, ...]", "DATA"), ...]
    >>> ml_mod.test_filter = [("FIELD_NAME", "FT:[==, ...]", "DATA"), ...]
    >>> # Import samples
    >>> ml_mod.sampleData(sd)
    >>> # Training
    >>> ml_mod.train()
    >>> # Save
    >>> ml_mod.save("*.shh2mod")
    >>> ml_mod.imdc("raster_fn")
    >>> # Load
    >>> ml_mod = MLModel().load("*.shh2mod")
    >>> ml_mod.imdc("raster_fn")

    """

    def __init__(self):
        super(MLModel, self).__init__()
        self.name = "MLModel"
        self.samples = _MLSamples()
        self.clf = None

    def sampleData(self, sd: SamplesData):
        self.samples = sd.ml(
            self.x_keys, self.map_dict, data_scale=self.data_scale,
            train_filters=self.train_filters, test_filter=self.test_filters,
            feat_funcs=self.feat_funcs,
        )

    def train(self, *args, **kwargs):
        x_train, y_train, x_test, y_test = self.samples.data()
        self.clf.fit(x_train, y_train)
        self.accuracy(x_test, y_test)

    def predict(self, x, *args, **kwargs):
        return self.clf.predict(x)

    def imdc(self, raster_fns, to_imdc_fn=None, *args, **kwargs):
        to_imdc_fn = _imdc1(
            self.filename, self, self.x_keys, raster_fns,
            self.data_scale, self.color_table, to_imdc_fn,
            feat_funcs=self.feat_funcs
        )
        return to_imdc_fn

    def save(self, filename=None, dirname=None, is_save_clf=True, is_save_data=True,is_samples=True, *args, **kwargs):
        filename = self._getfilename(dirname, filename)

        if is_save_clf:
            to_dirname = self._gettodirname(filename)
            to_clf_fn = os.path.join(to_dirname, "model.mod")
            joblib.dump(self.clf, to_clf_fn)

        if is_save_data:
            to_dirname = self._gettodirname(filename)
            to_data_fn = os.path.join(to_dirname, "{}_data.npy")
            np.save(to_data_fn.format("x_train"), np.array(self.samples.x_train))
            np.save(to_data_fn.format("x_test"), np.array(self.samples.x_test))
            np.save(to_data_fn.format("y_train"), np.array(self.samples.y_train))
            np.save(to_data_fn.format("y_test"), np.array(self.samples.y_test))

        to_dict = self.toDict(is_samples=is_samples)
        to_dict["__class__.__name__"] = self.__class__.__name__
        saveJson(to_dict, filename)

    def load(self, filename=None, dirname=None, *args, **kwargs):
        filename = self._getfilename(dirname, filename)
        if not os.path.isfile(filename):
            return None

        to_dirname = self._gettodirname(filename, False)
        to_dict = readJson(filename)
        self.loadDict(to_dict)

        to_clf_fn = os.path.join(to_dirname, "model.mod")
        if os.path.isfile(to_clf_fn):
            self.clf = joblib.load(to_clf_fn)

        def loaddata(_fn):
            if os.path.isfile(_fn):
                return np.load(_fn)
            else:
                return None

        to_data_fn = os.path.join(to_dirname, "{}_data.npy")
        self.samples.x_train = loaddata(to_data_fn.format("x_train"))
        self.samples.x_test = loaddata(to_data_fn.format("x_test"))
        self.samples.y_train = loaddata(to_data_fn.format("y_train"))
        self.samples.y_test = loaddata(to_data_fn.format("y_test"))
        return self

    def toDict(self, is_samples=True, *args, **kwargs):
        to_dict = super(MLModel, self).toDict()
        to_dict = {
            **to_dict,
            "train_filters": self.train_filters,
            "test_filter": self.test_filters,
        }
        if is_samples:
            to_dict["samples"] = self.samples.toDict()
        return to_dict

    def loadDict(self, to_dict):
        super(MLModel, self).loadDict(to_dict)
        self.train_filters = to_dict["train_filters"]
        self.test_filters = to_dict["test_filter"]
        self.samples = _MLSamples().loadDict(to_dict["samples"])

    def score(self, x=None, y=None, *args, **kwargs):
        if x is None:
            x = self.samples.x_test
            y = self.samples.y_test
        return accuracy_score(y, self.predict(x))


class TorchModel(_ModelInit):

    def __init__(self):
        super(TorchModel, self).__init__()

        self.name = "TorchModel"
        self.samples = _TorchSamples()

        self.model = None
        self.criterion = None
        self.win_size = None
        self.read_size = None

        self.epochs = 100
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.n_test = 10
        self.batch_size = 32
        self.batch_save = False
        self.epoch_save = True
        self.save_model_fmt = None
        self.n_epoch_save = 1

        self._optimizer = None
        self._scheduler = None

        def func_logit_category(model, x: torch.Tensor):
            with torch.no_grad():
                # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1182 {} >".format(x.shape))
                logit = model(x)
                # input(r"F:\PyCodes\SRTCodes\SRTModel.py::1184>")
                y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_epoch = None
        self.func_xy_deal = None
        self.func_batch = None
        self.func_loss_deal = None
        self.func_y_deal = lambda y: y + 1
        self.func_logit_category = func_logit_category
        self.func_print = print
        self.func_field_record_save = None

        self.n_imdc_one = -1

    def sampleData(self, sd: SamplesData):
        self.samples = sd.dltorch(
            self.map_dict, self.win_size, self.read_size,
            train_filters=self.train_filters, test_filter=self.test_filters,
            device=self.device,
        )
        self.x_keys = self.samples.keys
        self.fd = self.samples.fd
        return self

    def optimizer(self, optim_cls=optim.Adam, *args, **kwargs):
        """ optim.Adam lr=0.001, eps=0.0000001 """
        self._optimizer = optim_cls(self.model.parameters(), *args, **kwargs)

    def scheduler(self, scheduler_cls=optim.lr_scheduler.StepLR, *args, **kwargs):
        """ optim.lr_scheduler.StepLR step_size=20, gamma=0.6, last_epoch=-1 """
        self._scheduler = scheduler_cls(self._optimizer, *args, **kwargs)

    def train(self, *args, **kwargs):
        torch_training = TorchTraining()
        torch_training.model = self.model
        torch_training.criterion = self.criterion
        torch_training.epochs = self.epochs
        torch_training.device = self.device
        torch_training.n_test = self.n_test

        torch_training.trainLoader(self.samples.train_ds, batch_size=self.batch_size)
        torch_training.testLoader(self.samples.test_ds, batch_size=self.batch_size)

        torch_training.optimizer(optim.Adam, lr=0.001, eps=0.000001)
        torch_training.scheduler(optim.lr_scheduler.StepLR, step_size=20, gamma=0.6, last_epoch=-1)

        torch_training.initCM(cnames=self.cm_names)
        torch_training.batch_save = self.batch_save
        torch_training.epoch_save = self.epoch_save
        torch_training.save_model_fmt = self.save_model_fmt
        torch_training.n_epoch_save = self.n_epoch_save

        torch_training.func_epoch = self.func_epoch
        torch_training.func_xy_deal = self.func_xy_deal
        torch_training.func_batch = self.func_batch
        torch_training.func_loss_deal = self.func_loss_deal
        torch_training.func_y_deal = self.func_y_deal
        torch_training.func_logit_category = self.func_logit_category
        torch_training.func_print = self.func_print
        torch_training.func_field_record_save = self.func_field_record_save

        torch_training.train()
        return

    def imdc(self, raster_fns, to_imdc_fn=None, mod_fn=None, data_deal=None, read_size=(1000, -1),
             is_save_tiles=False, fun_print=print, *args, **kwargs):
        if mod_fn is not None:
            self.model.load_state_dict(torch.load(mod_fn))
            self.model.to(self.device)
        self.model.eval()
        self.model.zero_grad()

        def func_predict(x):
            return self.func_logit_category(self.model, x)

        if to_imdc_fn is None:
            if mod_fn is not None:
                to_imdc_fn = changext(mod_fn, "_imdc.tif")
            else:
                to_imdc_fn = changext(self.filename, "_imdc.tif")

        gti = GDALTorchImdc(raster_fns)
        gti.imdc3(
            func_predict=func_predict, win_size=self.win_size, to_imdc_fn=to_imdc_fn,
            fit_names=self.x_keys, data_deal=data_deal, color_table=self.color_table,
            is_jdt=True, device=self.device, read_size=read_size, is_save_tiles=is_save_tiles,
            fun_print=fun_print,
        )

        self.model.train()
        return to_imdc_fn

    def predict(self, x, *args, **kwargs):
        x = torch.from_numpy(x).to(self.device)
        y1 = self.func_logit_category(self.model, x)
        return y1.cpu().numpy()


def main():
    def func1():
        cm_names = ["IS", "VEG", "SOIL", "WAT"]

        sd = SamplesData()
        sd.addCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd5_spl.csv")

        ml_mod = MLModel()
        ml_mod.filename = r"F:\ProjectSet\Shadow\Hierarchical\Temp\tmp9.shh2mod"
        ml_mod.x_keys = _SHH2Config.NAMES
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
        ml_mod.data_scale.readJson(_SHH2Config.CD_RANGE_FN)
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
