# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTSample.py
@Time    : 2023/7/1 15:03
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of SRTSample
-----------------------------------------------------------------------------"""
import random

import numpy as np
import pandas as pd

from SRTCodes.SRTFeature import SRTFeatureCallBackCollection, SRTFeatureExtractionCollection
from SRTCodes.Utils import printList


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


def main():
    csv_spl = CSVSamples(r"F:\ProjectSet\Shadow\QingDao\Sample\qd_shadow_spl3_s1.csv")
    csv_spl.fieldNameCategory("CNAME")
    csv_spl.fieldNameTag("TAG")
    csv_spl.addCategoryNames(["NOT_KNOW", "IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH"])
    csv_spl.readData()

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

    csv_spl.print()


if __name__ == "__main__":
    main()
