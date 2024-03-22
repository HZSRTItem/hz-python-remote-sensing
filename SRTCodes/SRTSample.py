# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTSample.py
@Time    : 2023/7/1 15:03
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of SRTSample
-----------------------------------------------------------------------------"""
import csv
import json
import random

import numpy as np
import pandas as pd

from SRTCodes.SRTFeature import SRTFeatureCallBackCollection, SRTFeatureExtractionCollection
from SRTCodes.Utils import printList, SRTDataFrame, readJson, Jdt, SRTFilter


def filter_1(c_names, cate, select):
    if c_names is not None:
        for cname in c_names:
            select |= cate == cname
    return select


def getListNone(idx, list_iter):
    if len(list_iter) == 0:
        ret = None
    else:
        ret = list_iter[idx]
    return ret


class SRTFilterDF:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def filter(self, key, value):
        d = self.df[key].values
        self.df = self.df.loc[d == value]
        return self

    def get(self):
        return self.df


class Samples:

    def __init__(self):
        self._data = None
        self._category = None
        self._c_codes = None
        self._tag = None

        self._c_names: list = ["NOT_KNOW"]
        self._tag_names: list = []
        self._feat_names: list = []

        self._field_name_category = "CATEGORY"
        self._field_name_tag = "TAG"

        self._feat_callback_coll = SRTFeatureCallBackCollection(is_trans=True)
        self._feat_ext_coll = SRTFeatureExtractionCollection()

        self.feature_iter = []
        self.category_iter = []
        self.tag_iter = []

        self.is_trans = True
        self.is_scale_01 = True

        self.idx_f = 0
        self.idx_c = 0
        self.idx_t = 0
        self.is_end = False

        self.is_get_return_cname = False
        self.is_get_return_code_t = False

    def readData(self, *args, **kwargs):
        self._data = None

    def fieldNameCategory(self, field_name_category):
        self._field_name_category = field_name_category

    def fieldNameTag(self, field_name_tag):
        self._field_name_tag = field_name_tag

    def addCategoryName(self, cname: str):
        if cname not in self._c_names:
            self._c_names.append(cname)
        else:
            print("Warning: cname \"{0}\" in _c_names.".format(cname))

    def addCategoryNames(self, cnames: list):
        for cname in cnames:
            self.addCategoryName(cname)

    def addTag(self, tag: str):
        if tag not in self._tag_names:
            self._tag_names.append(tag)
        else:
            print("Warning: tag \"{0}\" in _tag_names.".format(tag))

    def addTags(self, tags: list):
        for tag in tags:
            self.addTag(tag)

    def _addCategoryList(self, c_arr: list):
        if self._category is None:
            category = []
        else:
            category = self._category.tolist()

        if self._c_codes is None:
            c_codes = []
        else:
            c_codes = self._c_codes.tolist()

        if self._tag is None:
            tag = []
        else:
            tag = self._tag.tolist()

        for cname in c_arr:
            if cname not in self._c_names:
                self._c_names.append(cname)
            category.append(cname)
            c_codes.append(self._c_names.index(cname))
            tag.append("")

        self._category = np.array(category)
        self._c_codes = np.array(c_codes)
        self._tag = np.array(tag)

    def _addTagList(self, tag_list):
        for i, tag in enumerate(tag_list):
            tag = str(tag)
            if tag not in self._tag_names:
                self._tag_names.append(tag)
            self._tag[i] = tag

    def getCategoryNames(self):
        return self._c_names.copy()

    def getFeatureNames(self):
        return self._feat_names.copy()

    def getTagNames(self):
        return self._tag_names.copy()

    def print(self):
        printList("Category Names:", self._c_names)
        printList("Feature Names:", self._feat_names)
        printList("Tag Names:", self._tag_names)

    def getFilter(self, *filters, **kwargs):
        df = self._data.copy()
        filter_df = SRTFilterDF(df)
        for k, v in filters:
            filter_df = filter_df.filter(k, v)
        for k in kwargs:
            filter_df = filter_df.filter(k, kwargs[k])
        return filter_df.get()

    def get(self, c_names=None, feat_names=None, tags=None):
        select1 = filter_1(c_names, self._category, np.zeros(len(self._category), dtype="bool"))
        if np.sum(select1) == 0:
            select1 = select1 == False
        select2 = filter_1(tags, self._tag, np.zeros(len(self._category), dtype="bool"))
        if np.sum(select2) == 0:
            select2 = select2 == False
        select = select1 & select2
        if feat_names is None:
            feat_names = self._feat_names
        d = self._data[feat_names][select]
        for k in d:
            if k in self._feat_callback_coll:
                d[k] = self._feat_callback_coll.fits(k, d[k])

        y = self._c_codes[select]

        if self.is_get_return_cname:
            y = self._category[select]

        if self.is_get_return_code_t:
            y_temp = self._category[select]
            for i, y_name in enumerate(y_temp):
                y[i] = c_names.index(y_name)

        return d, y

    def addFeatureIter(self, feat: list):
        for f in feat:
            if f not in self._feat_names:
                print("Warning: feature name \"{0}\" not in _feat_names.".format(f))
        self.feature_iter.append(feat)

    def addCategoryIter(self, cate: list):
        for f in cate:
            if f not in self._c_names:
                print("Warning: category name \"{0}\" not in _c_names.".format(f))
        self.category_iter.append(cate)

    def addTagIter(self, tag: list):
        for f in tag:
            if f not in self._tag_names:
                print("Warning: tag name \"{0}\" not in _c_names.".format(f))
        self.tag_iter.append(tag)

    def featureCallBack(self, feat_name, callback_func, is_trans=None):
        if is_trans is None:
            is_trans = self.is_trans
        self._feat_callback_coll.addCallBack(feat_name, callback_func=callback_func, is_trans=is_trans)

    def featureScaleMinMax(self, feat_name, x_min, x_max, is_trans=None, is_01=None):
        if is_trans is None:
            is_trans = self.is_trans
        if is_01 is None:
            is_01 = self.is_scale_01
        self._feat_callback_coll.addScaleMinMax(feat_name, x_min, x_max, is_trans=is_trans, is_to_01=is_01)

    def featureExtraction(self, feat_name, cal_feat_names, extract_func):
        self._feat_ext_coll.add(feat_name=feat_name, cal_feat_names=cal_feat_names, extract_func=extract_func)

    def isIn(self, name, value):
        """ c_name feat_name tag_name"""
        if name == "c_name":
            return value in self._c_names
        if name == "feat_name":
            return value in self._feat_names
        if name == "tag_name":
            return value in self._tag_names
        raise Exception("Name: c_name feat_name tag_name not " + name)

    def __iter__(self):
        return self

    def __next__(self):
        if self.is_end:
            raise StopIteration()
        c_names = getListNone(self.idx_c, self.category_iter)
        feat_names = getListNone(self.idx_f, self.feature_iter)
        tags = getListNone(self.idx_t, self.tag_iter)

        x, y = self.get(c_names=c_names, feat_names=feat_names, tags=tags)

        self.idx_c += 1
        if self.idx_c == len(self.category_iter) or len(self.category_iter) == 0:
            self.idx_c = 0
            self.idx_f += 1
            if self.idx_f == len(self.feature_iter) or len(self.feature_iter) == 0:
                self.idx_f = 0
                self.idx_t += 1
                if self.idx_t == len(self.tag_iter) or len(self.tag_iter) == 0:
                    self.idx_t = 0
                    self.is_end = True

        spl_dict = {"c_names": c_names, "feat_names": feat_names, "tags": tags}

        return x, y, spl_dict

    def __len__(self):
        return len(self._data)


class CSVSamples(Samples):
    """ category name, tag default kong"""

    def __init__(self, csv_fn=None, is_read=False):
        super(CSVSamples, self).__init__()
        self.csv_fn = csv_fn

        if self.csv_fn is not None:
            if is_read:
                self.readData()

    def readData(self, csv_fn=None):
        if csv_fn is not None:
            self.csv_fn = csv_fn
        df = pd.read_csv(self.csv_fn)
        for k in df:
            if k.lower() == self._field_name_category.lower():
                self._field_name_category = k
            elif k.lower() == self._field_name_tag.lower():
                self._field_name_tag = k
            else:
                self._feat_names.append(k)

        self._addCategoryList(df[self._field_name_category].values.tolist())
        if self._field_name_tag in df:
            self._addTagList(df[self._field_name_tag].values.tolist())
        self._data = df

    def saveToFile(self, csv_fn):
        self._data.to_csv(csv_fn, index=False)

    def getDFData(self):
        return self._data.copy()


class SRTSampleSelect:
    """
    输入每个类别抽样的个数和类别之间的映射
    """

    def __init__(self, x: pd.DataFrame = None, y=None, sampling_type="no_back"):
        """
        sampling_type: Is the sample a sample that has been returned or a sample that has not been returned `back|no_back`
        """
        self.x = x
        self.y = y
        self.data = {}
        self.sampling_type = sampling_type

        self.init()

    def init(self, x: pd.DataFrame = None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if (x is None) and (y is None):
            return

        for i in range(len(y)):
            y_tmp = int(y[i])
            if y_tmp not in self.data:
                self.data[y_tmp] = []
            self.data[y_tmp].append(x.loc[i].to_dict())

    def get(self, category_number_dict, map_dict=None):
        if map_dict is None:
            map_dict = {}
        out_df_list, out_y_list = [], []
        for category, number in category_number_dict.items():
            df_list = self.getByCategory(category, number)
            out_df_list += df_list
            if category in map_dict:
                category = map_dict[category]
            out_y_list += [category] * len(df_list)
        return pd.DataFrame(out_df_list), np.array(out_y_list)

    def getByCategory(self, category, number):
        number = min(number, len(self.data[category]))
        select_list = [i for i in range(len(self.data[category]))]
        random.shuffle(select_list)
        out_data_list = []
        if self.sampling_type == "back":
            for i in range(number):
                out_data_list.append(self.data[category][select_list[i]])
        elif self.sampling_type == "no_back":
            data = []
            for i in range(len(self.data[category])):
                d = self.data[category][select_list[i]]
                if i < number:
                    out_data_list.append(d)
                else:
                    data.append(d)
            self.data[category] = data
        return out_data_list

    def printNumber(self):
        for k in self.data:
            print("{0}:{1} ".format(k, len(self.data[k])), end="")
        print()


class SRTSample:

    def __init__(self, *field_datas, data=None, **field_kws):
        self.fields = {}
        self.data = data
        self._n_iter = 0
        self._keys = []

        self.name = None
        self.code = None
        self.x = None
        self.y = None
        self.srt = None

        self.init(field_datas, field_kws)

    def init(self, field_datas, field_kws):
        for field_data in field_datas:
            self.fields[field_data[0]] = field_data[1]
        for k in field_kws:
            self.fields[k] = field_kws[k]

    def keys(self):
        return list(self.fields.keys())

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return self

    def __next__(self):
        if self._n_iter == 0:
            self._keys = list(self.keys())
        if self._n_iter == len(self._keys):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._keys[self._n_iter - 1]

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, item):
        if isinstance(item, list):
            return [self.fields[i] for i in item]
        return self.fields[item]

    def __setitem__(self, key, value):
        self.fields[key] = value


class SRTSampleCollection:

    def __init__(self):
        self.samples = []
        self._n_iter = 0
        self.field_names = []

    def read_csv(self, csv_fn, field_datas=None, *args, **kwargs):
        if field_datas is None:
            field_datas = {}
        sdf = SRTDataFrame()
        sdf.read_csv(csv_fn, is_auto_type=True)
        for i in range(len(sdf)):
            spl = SRTSample()
            for k in sdf:
                spl[k] = sdf[k][i]
            for k in field_datas:
                spl[k] = field_datas[k]
            self.addSample(spl)

    def addSample(self, spl: SRTSample):
        for name in spl:
            self.addFieldName(name)
        self.samples.append(spl)

    def addFieldName(self, name):
        if name not in self.field_names:
            self.field_names.append(name)
        return name

    def keys(self):
        return self.field_names

    def getFields(self, *field_names):
        if len(field_names) == 0:
            to_field_names = self.keys()
        else:
            to_field_names = []
            for field_name in field_names:
                if isinstance(field_name, list) or isinstance(field_name, tuple):
                    to_field_names.extend(field_name)
                else:
                    to_field_names.append(field_name)
            if len(to_field_names) == 1:
                to_field_names = to_field_names[0]
        return [spl[to_field_names] for spl in self.samples]

    def setField(self, n_spl, field_name, data):
        self.addFieldName(field_name)
        self.samples[n_spl][field_name] = data
        return self.samples[n_spl]

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return self

    def __next__(self) -> SRTSample:
        if self._n_iter == len(self.samples):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self.samples[self._n_iter - 1]

    def __contains__(self, item):
        return item in self.samples

    def __getitem__(self, item) -> SRTSample:
        return self.samples[item]

    def __setitem__(self, key, value):
        self.samples[key] = value

    def copyNoSamples(self):
        ssc = SRTSampleCollection()
        return ssc

    def filter(self, _filter: SRTFilter, _scc=None):
        if _scc is None:
            _scc = self.copyNoSamples()
        for spl in self.samples:
            if _filter.compare(spl[_filter.compare_name]):
                _scc.addSample(spl)
        return _scc

    def filterFunc(self, _func, _scc=None):
        if _scc is None:
            _scc = self.copyNoSamples()
        for spl in self.samples:
            if _func(spl):
                _scc.addSample(spl)
        return _scc

    def filterCompare(self, _compare, _field_name, _compare_data, _scc=None):
        if _compare == "eq":
            return self.filter(_filter=SRTFilter.eq(_field_name, _compare_data), _scc=_scc)
        elif _compare == "lt":
            return self.filter(_filter=SRTFilter.lt(_field_name, _compare_data), _scc=_scc)
        elif _compare == "lte":
            return self.filter(_filter=SRTFilter.lte(_field_name, _compare_data), _scc=_scc)
        elif _compare == "gt":
            return self.filter(_filter=SRTFilter.gt(_field_name, _compare_data), _scc=_scc)
        elif _compare == "gte":
            return self.filter(_filter=SRTFilter.gte(_field_name, _compare_data), _scc=_scc)
        else:
            return _scc

    def map(self, func, is_ret_iter=False, *args, **kwargs):
        def _func(x):
            return func(x, *args, **kwargs)

        map_iter = map(_func, self.samples)
        if is_ret_iter:
            return map_iter
        for _ in map_iter:
            continue


class _Category:

    def __init__(self, name="NOT_KNOW", code=0, color=(0, 0, 0)):
        self.name = name
        self.code = code
        self.color = color

    def toDict(self):
        return {"NAME": self.name, "CODE": self.code, "COLOR": self.color}

    def loadJson(self, _dict):
        self.name = _dict["NAME"]
        self.code = _dict["CODE"]
        self.color = tuple(_dict["COLOR"])
        return self

    def copy(self):
        return _Category(name=self.name, code=self.code, color=self.color)

    def __str__(self):
        return "_Category(name={0}, code={1}, color={2})".format(self.name, self.code, self.color)


class _CategoryCollection:

    def __init__(self):
        self.coll = {}
        self._n_iter = 0
        self._keys = []

    def add(self, shh_category: _Category):
        if shh_category.name in self.coll:
            raise Exception("_Category of \"{0}\" have in collection.".format(shh_category.name))
        self.coll[shh_category.name] = shh_category

    def addDict(self, _dict):
        for name in _dict:
            self.add(_Category().loadJson(_dict[name]))

    def map(self, name):
        return self.coll[name].code

    def keys(self):
        return list(self.coll.keys())

    def __len__(self):
        return len(self.coll)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if self._n_iter == 0:
            self._keys = self.keys()
        if self._n_iter == len(self.coll):
            self._n_iter = 0
            raise StopIteration()
        self._n_iter += 1
        return self._keys[self._n_iter - 1]

    def __contains__(self, item):
        return item in self.coll

    def __getitem__(self, item) -> _Category:
        return self.coll[item]

    def toDict(self):
        return {k: self.coll[k].toDict() for k in self.coll}

    def copy(self):
        category_coll = _CategoryCollection()
        for k in self.coll:
            category_coll.add(self.coll[k].copy())
        return category_coll

    def __str__(self):
        to_str = "_CategoryCollection(\n"
        for k in self.coll:
            to_str += "    {0}={1},\n".format(k, self.coll[k])
        to_str += ")"
        return to_str

    def toCodeColor(self):
        return {cate.code: cate.color for cate in self.coll.values()}


def _sampleToDict(spl: SRTSample, is_save_data=False):
    if is_save_data:
        data = spl.data
        if isinstance(data, np.ndarray):
            data = data.tolist()
    else:
        data = None
    return {"NAME": spl.name, "CODE": spl.code, "X": spl.x, "Y": spl.y, "SRT": spl.srt, "FIELDS": spl.fields,
            "DATA": data}


class SRTCategorySampleCollection(SRTSampleCollection):

    def __init__(self):
        super(SRTCategorySampleCollection, self).__init__()
        self.category_coll = _CategoryCollection()

        self.FN_CNAME = "CNAME"
        self.FN_X = "X"
        self.FN_Y = "Y"
        self.FN_SRT = "SRT"

    def addCategory(self, name="NOT_KNOW", code=0, color=(0, 0, 0)):
        shh_category = _Category(name=name, code=code, color=color)
        self.category_coll.add(shh_category)

    def addSample(self, spl: SRTSample):
        for name in spl:
            self.addFieldName(name)
        if self.FN_CNAME in spl:
            spl.name = spl[self.FN_CNAME]
            spl.code = self.category_coll.map(spl.name)
        if self.FN_X in spl:
            spl.x = float(spl[self.FN_X])
        if self.FN_Y in spl:
            spl.y = float(spl[self.FN_Y])
        if self.FN_SRT in spl:
            spl.srt = int(spl[self.FN_SRT])
        self.samples.append(spl)

    def toJson(self, json_fn, is_save_data=False, is_jdt=False):
        with open(json_fn, "w", encoding="utf-8") as f:
            f.write("{")
            f.write("\"FN_CNAME\":\"{0}\",".format(self.FN_CNAME))
            f.write("\"FN_X\":\"{0}\",".format(self.FN_X))
            f.write("\"FN_Y\":\"{0}\",".format(self.FN_Y))
            f.write("\"FN_SRT\":\"{0}\",".format(self.FN_SRT))
            f.write("\"CATEGORY_COLL\":")
            json.dump(self.category_coll.toDict(), f)
            f.write(",")
            f.write("\"SAMPLES\":[")
            jdt = Jdt(len(self.samples), "ShadowHierarchicalSampleCollection::toJson")
            if is_jdt:
                jdt.start()
            for i, spl in enumerate(self.samples):
                json.dump(_sampleToDict(spl, is_save_data), f)
                if i != (len(self.samples) - 1):
                    f.write(",")
                if is_jdt:
                    jdt.add()
            if is_jdt:
                jdt.end()
            f.write("]")
            f.write("}")

    def readJson(self, json_fn):
        json_dict = readJson(json_fn)
        self.FN_CNAME = json_dict["FN_CNAME"]
        self.FN_X = json_dict["FN_X"]
        self.FN_Y = json_dict["FN_Y"]
        self.FN_SRT = json_dict["FN_SRT"]
        self.category_coll.addDict(json_dict["CATEGORY_COLL"])
        for spl_dict in json_dict["SAMPLES"]:
            if spl_dict["DATA"] is not None:
                spl_dict["DATA"] = np.array(spl_dict["DATA"])
            spl = SRTSample(data=spl_dict["DATA"])
            spl.name = spl_dict["NAME"]
            spl.code = spl_dict["CODE"]
            spl.x = spl_dict["X"]
            spl.y = spl_dict["Y"]
            spl.srt = spl_dict["SRT"]
            spl.fields = spl_dict["FIELDS"]
            self.addSample(spl)

    def toCSV(self, csv_fn):
        with open(csv_fn, "w", encoding="utf-8", newline="") as f:
            cw = csv.writer(f)
            names = self.field_names.copy()
            if self.FN_CNAME in names:
                if "__CODE__" not in names:
                    names.append("__CODE__")
            cw.writerow(names)
            for i in range(len(self.samples)):
                spl: SRTSample = self.samples[i]
                fields = spl.fields.copy()
                if self.FN_CNAME in spl:
                    fields[self.FN_CNAME] = spl.name
                    fields["__CODE__"] = spl.code
                if self.FN_X in spl:
                    fields[self.FN_X] = spl.x
                if self.FN_Y in spl:
                    fields[self.FN_Y] = spl.y
                if self.FN_SRT in spl:
                    fields[self.FN_SRT] = spl.srt
                cw.writerow(list(fields.values()))

    def copyNoSamples(self):
        scsc = SRTCategorySampleCollection()
        self.copySCSC(scsc)
        return scsc

    def copySCSC(self, scsc):
        scsc.category_coll = self.category_coll.copy()
        scsc.FN_CNAME = self.FN_CNAME
        scsc.FN_X = self.FN_X
        scsc.FN_Y = self.FN_Y
        scsc.FN_SRT = self.FN_SRT

    def saveDataToNPY(self, npy_fn, dtype=None):
        # spl_data = np.zeros(
        #     (len(self.samples), self.samples[0].data.shape[0],
        #      self.samples[0].data.shape[1], self.samples[0].data.shape[2],))
        spl_data = []
        for i, spl in enumerate(self.samples):
            # spl_data[i, :, :] = spl.data
            spl_data.append(np.array([spl.data]))
        spl_data = np.concatenate(spl_data)
        if dtype is not None:
            spl_data = spl_data.astype(dtype)
        np.save(npy_fn, spl_data)

    def loadDataFromNPY(self, npy_fn, dtype=None):
        spl_data = np.load(npy_fn)
        if dtype is not None:
            spl_data = spl_data.astype(dtype)
        for i, spl in enumerate(self.samples):
            spl.data = spl_data[i]


class SRTCSplColl(SRTCategorySampleCollection):

    def __init__(self):
        super(SRTCSplColl, self).__init__()


def main():
    # csv_spl = CSVSamples(r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_spl3_s1.csv")
    # csv_spl.fieldNameCategory("CNAME")
    # csv_spl.fieldNameTag("TAG")
    # csv_spl.addCategoryNames(["NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"])
    # csv_spl.readData()

    # csv_spl.is_get_return_cname = True
    # csv_spl.is_get_return_code_t = True
    # d = csv_spl.get(feat_names=["B", "G", "R", "N", "VV_AS", "VH_AS", "VV_DE"],
    #                 c_names=["NOT_KNOW", "SOIL", "VEG"])

    # csv_spl.addCategoryIter(["IS", "VEG"])
    # csv_spl.addCategoryIter(["SOIL", "VEG"])
    # csv_spl.addCategoryIter(["WATER", "VEG"])

    # csv_spl.addFeatureIter(["X", "Y", "SRT", "B", "G", "R", "N", "VV_AS", "VH_AS", "VV_DE"])
    # csv_spl.addFeatureIter(["VH_DE", "NDVI", "NDWI", "Optical_PC1_Variance", "Optical_PC1_Homogeneity"])
    # csv_spl.addFeatureIter(["Optical_PC1_Contrast", "VH_DE_Mean", "VH_DE_Variance", "VH_DE_Homogeneity", "VV_DE_Mean"])
    # csv_spl.addFeatureIter(["VV_DE_Variance", "VV_DE_Homogeneity", "VH_AS_Mean", "VH_AS_Variance"])
    # csv_spl.addFeatureIter(["VH_AS_Homogeneity", "VV_AS_Mean", "VV_AS_Variance", "VV_AS_Homogeneity"])
    # csv_spl.addFeatureIter(["DE_20210430_C22", "DE_20210430_C12real", "DE_20210430_C12imag", "DE_20210430_C11"])
    # csv_spl.addFeatureIter(["AS_20210507_C22", "AS_20210507_C12real", "AS_20210507_C12imag", "AS_20210507_C11"])

    # for i, (x, y, d) in enumerate(csv_spl):
    #     print(i, x.shape, y.shape, d)

    # csv_spl.print()

    ssc = SRTSampleCollection()
    ssc.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\20231209\20240105H205307\20240105H205307_train_spl.csv")
    d = ssc.getFields()

    pass


if __name__ == "__main__":
    main()
