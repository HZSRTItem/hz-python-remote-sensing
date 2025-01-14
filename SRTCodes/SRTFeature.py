# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTFeature.py
@Time    : 2023/7/7 10:30
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of SRTFeature
-----------------------------------------------------------------------------"""
import warnings

import numpy as np
import pandas as pd

from SRTCodes.SRTCollection import SRTCollection
from SRTCodes.Utils import printKeyValue


def initCallBack(x):
    return x


class SRTFeatureCallBack:

    def __init__(self, callback_func=None, is_trans=False, feat_name=None):
        if callback_func is None:
            callback_func = initCallBack
        self.call_back = callback_func
        self.is_trans = is_trans
        self.feat_name = feat_name

    def fit(self, x: np.ndarray, *args, **kwargs):
        if self.is_trans:
            x = self.call_back(x)
        return x


class SRTFeatureCallBackScaleMinMax(SRTFeatureCallBack):

    def __init__(self, x_min=None, x_max=None, is_trans=False, is_to_01=False):
        super(SRTFeatureCallBackScaleMinMax, self).__init__(is_trans=is_trans)
        self.x_min = x_min
        self.x_max = x_max
        self.is_to_01 = is_to_01

    def fit(self, x: np.ndarray, *args, **kwargs):
        if self.is_trans:
            x_min, x_max = self.x_min, self.x_max
            if x_min is None:
                x_min = np.min(x)
            if x_max is None:
                x_max = np.max(x)
            x = np.clip(x, x_min, x_max)
            if self.is_to_01:
                x = (x - x_min) / (x_max - x_min)
        return x


class SRTFeatureCallBackList:

    def __init__(self):
        self._callback_list = []
        self._n_iter = 0

    def addScaleMinMax(self, x_min=None, x_max=None, is_trans=False, is_to_01=False):
        self._callback_list.append(SRTFeatureCallBackScaleMinMax(
            x_min=x_min, x_max=x_max, is_trans=is_trans, is_to_01=is_to_01))
        return self

    def add(self, callback_func, is_trans=False):
        self._callback_list.append(SRTFeatureCallBack(callback_func=callback_func, is_trans=is_trans))

    def __len__(self):
        return len(self._callback_list)

    def __getitem__(self, item):
        return self._callback_list[item]

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == len(self._callback_list):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._callback_list[self._n_iter - 1]

    def __contains__(self, item):
        return item in self._callback_list

    def fit(self, x):
        for callback in self._callback_list:
            x = callback.fit(x)
        return x


class SRTFeatureCallBackCollection(SRTCollection):

    def __init__(self, is_trans=False):
        super(SRTFeatureCallBackCollection, self).__init__()
        self.is_trans = is_trans
        self._feat_callbacks = {}

    def addScaleMinMax(self, feat_name, x_min=None, x_max=None, is_trans=False, is_to_01=False):
        callback_ = SRTFeatureCallBackScaleMinMax(x_min=x_min, x_max=x_max, is_trans=is_trans, is_to_01=is_to_01)
        self._addCallBack(feat_name, callback_)

    def addCallBack(self, feat_name: str, callback_func, is_trans=False):
        callback_ = SRTFeatureCallBack(callback_func=callback_func, is_trans=is_trans)
        self._addCallBack(feat_name, callback_)

    def _addCallBack(self, feat_name: str, callback_):
        if feat_name not in self._feat_callbacks:
            self._feat_callbacks[feat_name] = []
            self._n_next.append(feat_name)
        self._feat_callbacks[feat_name].append(callback_)

    def __getitem__(self, feat_name):
        return self._feat_callbacks[feat_name]

    def fits(self, feat_name, x):
        for call_back in self._feat_callbacks[feat_name]:
            x = call_back.fit(x)
        return x


class SRTFeatureExtraction(SRTCollection):

    def __init__(self, new_feat_name="FEATURE", cal_feat_names=None, extract_func=None):
        super().__init__()

        if cal_feat_names is None:
            cal_feat_names = []

        self.new_feat_name = new_feat_name
        self.features = {}
        self.extract_func = extract_func

        for cal_feat_name in cal_feat_names:
            self.addFeatureData(cal_feat_name, feat_data=None)

    def fit(self, *args, **kwargs):
        return self.extract_func(self.features, args=args, kwargs=kwargs)

    def addFeatureData(self, feat_name, feat_data=None):
        if feat_name not in self.features:
            self._n_next.append(feat_name)
        self.features[feat_name] = feat_data

    def __getitem__(self, feat_name):
        return self.features[feat_name]

    def __setitem__(self, feat_name, feat_data):
        self.addFeatureData(feat_name, feat_data)


class SRTFeatExtTwo(SRTFeatureExtraction):

    def __init__(self, new_feat_name, feat_name_first, feat_name_second, extract_func):
        super(SRTFeatExtTwo, self).__init__(
            new_feat_name=new_feat_name, cal_feat_names=[feat_name_first, feat_name_second], extract_func=extract_func)

    def fit(self, *args, **kwargs):
        d_keys = list(self.features.keys())
        x = self.extract_func(self.features[d_keys[0]], self.features[d_keys[1]])
        return x


def _extFunc_NormalizedDifference(x1, x2):
    return (x1 - x2) / (x1 + x2)


def _extFunc_Add(x1, x2):
    return x1 + x2


def _extFunc_Subtract(x1, x2):
    return x1 - x2


def _extFunc_Multiply(x1, x2):
    return x1 * x2


def _extFunc_Divide(x1, x2):
    return x1 / x2


class SRTFeatureExtractionCollection(SRTCollection):

    def __init__(self):
        super(SRTFeatureExtractionCollection, self).__init__()
        self.feat_extractions = {}
        self.o_feat_names = None
        self.o_data = None

    def add(self, new_feat_name, cal_feat_names, extract_func):
        self._checkFeatureName(new_feat_name)
        self.feat_extractions[new_feat_name] = SRTFeatureExtraction(
            new_feat_name=new_feat_name, cal_feat_names=cal_feat_names, extract_func=extract_func)

    def addTwo(self, new_feat_name, ext_func, feat_name_first, feat_name_second):
        self._checkFeatureName(new_feat_name)
        self.feat_extractions[new_feat_name] = SRTFeatExtTwo(
            new_feat_name=new_feat_name, feat_name_first=feat_name_first, feat_name_second=feat_name_second
            , extract_func=ext_func)

    def addNormalizedDifference(self, new_feat_name, feat_name_first, feat_name_second):
        self.addTwo(new_feat_name=new_feat_name,
                    ext_func=_extFunc_NormalizedDifference,
                    feat_name_first=feat_name_first,
                    feat_name_second=feat_name_second)

    def addAdd(self, new_feat_name, feat_name_first, feat_name_second):
        self.addTwo(new_feat_name=new_feat_name,
                    ext_func=_extFunc_Add,
                    feat_name_first=feat_name_first,
                    feat_name_second=feat_name_second)

    def addSubtract(self, new_feat_name, feat_name_first, feat_name_second):
        self.addTwo(new_feat_name=new_feat_name,
                    ext_func=_extFunc_Subtract,
                    feat_name_first=feat_name_first,
                    feat_name_second=feat_name_second)

    def addMultiply(self, new_feat_name, feat_name_first, feat_name_second):
        self.addTwo(new_feat_name=new_feat_name,
                    ext_func=_extFunc_Multiply,
                    feat_name_first=feat_name_first,
                    feat_name_second=feat_name_second)

    def addDivide(self, new_feat_name, feat_name_first, feat_name_second):
        self.addTwo(new_feat_name=new_feat_name,
                    ext_func=_extFunc_Divide,
                    feat_name_first=feat_name_first,
                    feat_name_second=feat_name_second)

    def _checkFeatureName(self, new_feat_name):
        if new_feat_name in self.feat_extractions:
            raise Exception("Feature name \"{0}\" have in".format(new_feat_name))

    def fits(self, o_feat_names, o_data):
        self._initFeatData(o_data, o_feat_names)
        ret_names = o_feat_names.copy()
        ret_d = o_data
        for feat_ext in self.feat_extractions:
            for fn in feat_ext:
                feat_ext[fn] = self._getData(fn)
            ret_names.append(feat_ext.new_feat_name)
            ret_d.append(feat_ext.fit())
        ret_d = np.concatenate(ret_d)
        return ret_names, ret_d

    def _initFeatData(self, o_data, o_feat_names):
        self.o_feat_names = {}
        self.o_data = o_data
        for i, feat_name in enumerate(o_feat_names):
            self.o_feat_names[feat_name] = i

    def _getData(self, feat_name):
        return self.o_data[self.o_feat_names[feat_name], :]


class SRTFeatureForward:

    def __init__(self, name):
        self.name = name


class SRTFeatureData:

    def __init__(self, name="FD", _data=None):
        self._name = name
        self._data = None
        self.data(_data)
        self.callback_list = SRTFeatureCallBackList()

    def data(self, _data=None):
        if _data is not None:
            self._data = _data
        return self._data

    def addCallBack(self, callback_func, is_trans=False):
        self.callback_list.add(callback_func=callback_func, is_trans=is_trans)
        self._data = callback_func(self._data)

    def scaleMinMax(self, x_min=None, x_max=None, is_to_01=False):
        sfcbsmm = SRTFeatureCallBackScaleMinMax(x_min=x_min, x_max=x_max, is_trans=True, is_to_01=is_to_01)
        self._data = sfcbsmm.fit(self._data)
        return self

    def add(self, number):
        self._data = self._data + number
        return self

    def subtract(self, number):
        self._data = self._data - number
        return self

    def multiply(self, number):
        self._data = self._data * number
        return self

    def divide(self, number):
        self._data = self._data / number
        return self

    def funcCal(self, cal_func):
        self._data = cal_func(self._data)


class SRTFeatureDataCollection(SRTCollection):
    """ Feature Data Collection"""

    INIT_FEAT_NAME = "FEATURE"

    def __init__(self):
        super(SRTFeatureDataCollection, self).__init__()
        self._collection = {}
        self._n_next = self._collection

    def add(self, name=None, data=None):
        if (name is None) and (data is None):
            return None
        if name is None:
            n = 1
            while True:
                name = self.INIT_FEAT_NAME + "_{0}".format(n)
                if name not in self._collection:
                    break
        self._collection[name] = SRTFeatureData(name=name, _data=data)

    def addDataFrame(self, df):
        for k in df:
            self.add(k, df[k].values)

    def addCSVFile(self, csv_fn):
        self.addDataFrame(pd.read_csv(csv_fn))

    def addExcelFile(self, excel_fn, sheet_name=0):
        self.addDataFrame(pd.read_excel(excel_fn, sheet_name=sheet_name))

    def get(self, name):
        return self._collection[name].data()

    def __getitem__(self, feat_name) -> SRTFeatureData:
        return self._n_next[feat_name]

    def __setitem__(self, feat_name, feat_data):
        self._n_next[feat_name] = feat_data

    def keys(self):
        return self._collection.keys()


class SRTFeaturesForwards:

    def __init__(self):
        self._init_forward_name = "FORWARD"
        self.feature_forwards = {}

    def addForwardCallBackObj(self, callback_obj: SRTFeatureCallBack, feat_name: str = None, forward_name: str = None):
        forward_name = self._getForwardName(forward_name)
        callback_obj.feat_name = feat_name
        self.feature_forwards[forward_name] = {"type": "callback", "callback_obj": callback_obj}

    def addForwardCallBack(self, feat_name: str, callback_func, forward_name=None):
        callback_ = SRTFeatureCallBack(callback_func=callback_func, is_trans=True, feat_name=feat_name)
        self.addForwardCallBackObj(callback_obj=callback_, feat_name=feat_name, forward_name=forward_name)
        # np.clip(callback_func)

    def addFCBScaleMinMax(self, feat_name: str, x_min: float = None, x_max: float = None, is_to_01: bool = False,
                          forward_name: str = None):
        callback_ = SRTFeatureCallBackScaleMinMax(x_min=x_min, x_max=x_max, is_trans=True, is_to_01=is_to_01)
        self.addForwardCallBackObj(callback_obj=callback_, feat_name=feat_name, forward_name=forward_name)

    def addForwardFeatureExtractionObj(self, feat_ext_obj: SRTFeatureExtraction, forward_name: str = None):
        forward_name = self._getForwardName(forward_name)
        self.feature_forwards[forward_name] = {"type": "extraction", "extraction_obj": feat_ext_obj}

    def addForwardFeatureExtraction(self, new_feat_name: str, cal_feat_names: str, extract_func,
                                    forward_name: str = None):
        feat_ext = SRTFeatureExtraction(new_feat_name=new_feat_name, cal_feat_names=cal_feat_names,
                                        extract_func=extract_func)
        self.addForwardFeatureExtractionObj(feat_ext_obj=feat_ext, forward_name=forward_name)

    def addFFETwo(self, new_feat_name, ext_func, feat_name_first, feat_name_second, forward_name=None):
        feat_ext = SRTFeatExtTwo(new_feat_name=new_feat_name,
                                 feat_name_first=feat_name_first,
                                 feat_name_second=feat_name_second,
                                 extract_func=ext_func)
        self.addForwardFeatureExtractionObj(feat_ext_obj=feat_ext, forward_name=forward_name)

    def addFFENormalizedDifference(self, new_feat_name, feat_name_first, feat_name_second, forward_name=None):
        self.addFFETwo(new_feat_name=new_feat_name,
                       ext_func=_extFunc_NormalizedDifference,
                       feat_name_first=feat_name_first,
                       feat_name_second=feat_name_second,
                       forward_name=forward_name)

    def addFFEAdd(self, new_feat_name, feat_name_first, feat_name_second, forward_name=None):
        self.addFFETwo(new_feat_name=new_feat_name,
                       ext_func=_extFunc_Add,
                       feat_name_first=feat_name_first,
                       feat_name_second=feat_name_second,
                       forward_name=forward_name)

    def addFFESubtract(self, new_feat_name, feat_name_first, feat_name_second, forward_name=None):
        self.addFFETwo(new_feat_name=new_feat_name,
                       ext_func=_extFunc_Subtract,
                       feat_name_first=feat_name_first,
                       feat_name_second=feat_name_second,
                       forward_name=forward_name)

    def addFFEMultiply(self, new_feat_name, feat_name_first, feat_name_second, forward_name=None):
        self.addFFETwo(new_feat_name=new_feat_name,
                       ext_func=_extFunc_Multiply,
                       feat_name_first=feat_name_first,
                       feat_name_second=feat_name_second,
                       forward_name=forward_name)

    def addFFEDivide(self, new_feat_name, feat_name_first, feat_name_second, forward_name=None):
        self.addFFETwo(new_feat_name=new_feat_name,
                       ext_func=_extFunc_Divide,
                       feat_name_first=feat_name_first,
                       feat_name_second=feat_name_second,
                       forward_name=forward_name)

    def _getForwardName(self, forward_name):
        is_find = False
        init_fn = self._init_forward_name
        if forward_name is None:
            is_find = True
        if forward_name in self.feature_forwards:
            init_fn = forward_name
            is_find = True
        if is_find:
            forward_ = 1
            while True:
                forward_name = "{0}_{1}".format(init_fn, forward_)
                forward_ += 1
                if forward_name not in self.feature_forwards:
                    break
        return forward_name


class SRTFeatures(SRTFeaturesForwards):

    def __init__(self):
        super().__init__()
        self.features = {}
        self.data_shape = None
        self._n_next = []

    def addFeature(self, feat_name: str, data: np.ndarray = None):
        if len(self.features) != 0:
            if feat_name in self.features:
                raise Exception("Feature \"{0}\" have in features and change feature name.".format(feat_name))
            if data is not None:
                if self.data_shape is not None:
                    if data.shape != self.data_shape:
                        warnings.warn("data shape {0} can not equal this {1}".format(data.shape, self.data_shape))
        self.features[feat_name] = data
        if self.data_shape is None:
            if data is not None:
                self.data_shape = data.shape
        self._n_next.append(feat_name)

    def forward(self):
        for i, forward_name in self.feature_forwards:
            # self.feature_forwards[forward_name] = {"type": "callback", "callback_obj": callback_obj}
            # self.feature_forwards[forward_name] = {"type": "extraction", "extraction_obj": feat_ext_obj}
            forward_args = self.feature_forwards[forward_name]
            print("{0:>2d}. {1}[{2}]".format(i + 1, forward_name, forward_args["type"]))

            if forward_args["type"] == "callback":
                callback_obj: SRTFeatureCallBack = forward_args["callback_obj"]
                self.features[callback_obj.feat_name] = callback_obj.fit(self.features[callback_obj.feat_name])

                printKeyValue("TYPE", callback_obj.__class__.__name__)
                printKeyValue("FEATURE NAME", callback_obj.feat_name)

            elif forward_args["type"] == "extraction":
                extraction_obj: SRTFeatureExtraction = forward_args["extraction_obj"]
                for feat_name in extraction_obj:
                    extraction_obj[feat_name] = self.features[feat_name]
                self.features[extraction_obj.new_feat_name] = extraction_obj.fit()

                printKeyValue("TYPE", extraction_obj.__class__.__name__)
                printKeyValue("NEW FEATURE NAME", extraction_obj.new_feat_name)
                printKeyValue("CAL FEATURE NAME", ", ".join(extraction_obj.features.keys()))

    def _addCallBackCheck(self, feat_name):
        if feat_name not in self.features:
            raise Exception("Feature \"{0}\" have not in features.".format(feat_name))
        return feat_name

    def __getitem__(self, feat_name_or_number):
        if isinstance(feat_name_or_number, int):
            feat_name_or_number = self._n_next[feat_name_or_number]
        return self.features[feat_name_or_number]

    def __setitem__(self, feat_name_or_number, arr):
        if isinstance(feat_name_or_number, int):
            feat_name_or_number = self._n_next[feat_name_or_number]
        self.features[feat_name_or_number] = arr


class SRTFeaturesMemory:

    def __init__(self, length=None, names=None):
        self.names = []
        self._callbacks = []
        self.length = length
        if length is None:
            if names is not None:
                self.length = len(names)
                self.names = names
        else:
            self.length = length
            self.initNames()

    def __len__(self):
        return self.length

    def initNames(self, names=None):
        if names is None:
            names = ["Name{}".format(i) for i in range(self.length)]
        else:
            if len(names) != self.length:
                warnings.warn("length of names not eq data. {} not eq {}".format(len(names), self.length))
        self.names = names
        return self

    def initCallBacks(self):
        self._callbacks = [SRTFeatureCallBackList() for _ in range(self.length)]
        return self

    def callbacks(self, name_number) -> SRTFeatureCallBackList:
        return self._callbacks[self._name_number(name_number)]

    def _name_number(self, name_number):
        if isinstance(name_number, str):
            return self.names.index(name_number)
        else:
            return name_number


class SRTFeaturesCalculation:
    """ Features calculate self, extraction

    type: data type as df|np
    """

    DATA_TYPES = ["df", "np"]

    def __init__(self, *init_names):
        self.init_names = list(init_names)
        self.type = None
        self.data = None
        self.calculation = []

    def initData(self, _type, _data):
        if _type not in self.DATA_TYPES:
            raise Exception("Can not support data type of \"{}\". Not in {}.".format(_type, self.DATA_TYPES))
        self.type = _type
        self.data = _data

    def __getitem__(self, item):
        if self.type == "np":
            return self.data[self.init_names.index(item)]
        elif self.type == "df":
            return self.data[item]
        else:
            return None

    def __setitem__(self, key, value):
        if self.type == "np":
            self.data[self.init_names.index(key)] = value
        elif self.type == "df":
            self.data[key] = value

    def __contains__(self, item):
        return item in self.init_names

    def add(self, init_name, fit_names, func):
        if init_name not in self.init_names:
            self.init_names.append(init_name)
            warnings.warn("Init name \"{}\" not in list and add.".format(init_name))
        self.calculation.append((init_name, fit_names, func))

    def fit(self, ):
        for init_name, fit_names, func in self.calculation:
            datas = {}
            for name in fit_names:
                if self.type == "df":
                    datas[name] = self.__getitem__(name).values
                else:
                    datas[name] = self.__getitem__(name)
            data = func(datas)
            self.__setitem__(init_name, data)



def main():
    pass


if __name__ == "__main__":
    main()
