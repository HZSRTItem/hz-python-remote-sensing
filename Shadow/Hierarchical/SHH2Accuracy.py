# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Accuracy.py
@Time    : 2024/6/7 9:27
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Accuracy
-----------------------------------------------------------------------------"""
import os.path
import random
import time

import numpy as np
import pandas as pd

from SRTCodes.GDALUtils import GDALSamplingImageClassification
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.TrainingUtils import SRTAccuracy
from SRTCodes.Utils import FN, TimeName, FRW, DirFileName
from Shadow.Hierarchical.SHH2ML2 import SHH2ML_TrainImdc
from Shadow.Hierarchical.SHH2MLModel import mapDict
from Shadow.Hierarchical.SHH2Sample import SHH2SamplesManage


def sampleFC(y1, y2, category, is_eq_number=True):
    if not (isinstance(category, list) or isinstance(category, tuple)):
        category = [category]
    category = list(category)
    n_c = 0
    i_select_c1 = []
    i_select_c2 = []
    for i, y in enumerate(y1):
        if y in category[0]:
            n_c += 1
            i_select_c1.append(i)
        elif y in category[1]:
            i_select_c2.append(i)
    random.shuffle(i_select_c1)
    random.shuffle(i_select_c2)

    if is_eq_number:
        if len(i_select_c1) < len(i_select_c2):
            i_select = i_select_c1 + i_select_c2[:len(i_select_c1)]
        else:
            i_select = i_select_c1[:len(i_select_c2)] + i_select_c2
    else:
        i_select = i_select_c1 + i_select_c2

    i_select.sort()
    y1_tmp, y2_tmp = [], []
    for i in i_select:
        y1_tmp.append(y1[i])
        y2_tmp.append(y2[i])

    return y1_tmp, y2_tmp


def accuracyY12(y1, y2, y1_map_dict, y2_map_dict, cnames, fc_category=None, is_eq_number=True):
    sa = SRTAccuracy()
    sa.y1 = y1
    sa.y1_map_dict = y1_map_dict
    sa.y2 = y2
    sa.y2_map_dict = y2_map_dict
    if fc_category is not None:
        # _y1, _y2 = sa.categoryMap(sa.y1, sa.y1_map_dict), sa.categoryMap(sa.y2, sa.y2_map_dict)
        _y1, _y2 = sa.y1, sa.y2
        sa.y1, sa.y2 = sampleFC(_y1, _y2, fc_category, is_eq_number)
    sa.cnames = cnames
    cm = sa.cm()
    return cm


class SHH2AccTiao(SHH2SamplesManage):

    def __init__(self, *csv_fns):
        super().__init__()
        self.addCSVS(*csv_fns)
        self.cnames = []
        self.y2_map_dict = {}
        self.y1_map_dict = {}
        self.gsic = GDALSamplingImageClassification()

    def addGSICDirectory(self, dirname, fns):
        for fn in fns:
            self.gsic.add(fn, os.path.join(dirname, fn + "_imdc.tif"))
            print(os.path.join(dirname, fn + "_imdc.tif"))

    def addGSICRaster(self, fn, raster_fn):
        self.gsic.add(fn, raster_fn)

    def accuracy(self, gsic_name, y1_map_dict=None, y2_map_dict=None, cnames=None, fc_category=None, is_eq_number=True):
        if cnames is None:
            cnames = self.cnames
        if y2_map_dict is None:
            y2_map_dict = self.y2_map_dict
        if y1_map_dict is None:
            y1_map_dict = self.y1_map_dict

        y1 = self.c_list
        gsic = self.gsic[gsic_name]
        x_list, y_list = self.x_list, self.y_list
        y2 = gsic.sampling(x_list, y_list)

        cm = accuracyY12(y1, y2, y1_map_dict, y2_map_dict, cnames, fc_category, is_eq_number=is_eq_number)

        print(cm.fmtCM())
        is_cm = cm.accuracyCategory("IS")
        print(is_cm.fmtCM())
        return {
            "NAME": gsic_name,
            "IS_OA": is_cm.OA(), "IS_Kappa": is_cm.getKappa(),
            "OA": cm.OA(), "Kappa": cm.getKappa(),
        }

    def tacc1(self, fc_category=None):
        to_list = []
        for k in self.gsic.keys():
            to_list.append(self.accuracy(k, fc_category=fc_category))
        print(pd.DataFrame(to_list).sort_values("IS_OA", ascending=False))
        return to_list

    def saveAccuracy(self, dirname, name, **kwargs):
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        tn = TimeName(_dirname=dirname)
        to_csv_fn = tn.filename("{}_" + name + ".csv")
        self.toCSV(to_csv_fn)
        to_json_fn = tn.filename("{}_" + name + ".json")
        print("to_csv_fn", to_csv_fn)
        print("to_json_fn", to_json_fn)

        to_dict = {
            **kwargs,
            "gsic": self.gsic.toDict(),
            "spl_fns": self.spl_fns,
            "cnames": self.cnames,
            "y2_map_dict": self.y2_map_dict,
            "y1_map_dict": self.y1_map_dict,
            "x_field_name": self.x_field_name,
            "y_field_name": self.y_field_name,
            "c_field_name": self.c_field_name,
            "x_list": self.x_list,
            "y_list": self.y_list,
            "c_list": self.c_list,
        }
        FRW(to_json_fn).saveJson(to_dict)


class Citys:

    def __init__(self):
        self.qd = None
        self.bj = None
        self.cd = None

        self._next_qd = False
        self._next_bj = False
        self._next_cd = False

    def __len__(self):
        return 3

    def __iter__(self):
        return self

    def __next__(self):
        if not self._next_qd:
            self._next_qd = True
            return self.qd
        elif not self._next_bj:
            self._next_bj = True
            return self.bj
        elif not self._next_cd:
            self._next_cd = True
            return self.cd
        else:
            self._next_qd = False
            self._next_bj = False
            self._next_cd = False
            raise StopIteration()

    def __contains__(self, item):
        return item in [self.qd, self.bj, self.cd]


class SHH2ML_TST_Three_Main:

    def __init__(self):
        self.csv_fn = ""
        self.td = None
        self.dfn = None
        self.sh2_tsts = Citys()
        self.dfns = Citys()
        self.oa_kappa_cms = Citys()
        self.oa_kappa_IO_cms = Citys()
        self.oa_kappa_IS_cms = Citys()
        self.oa_kappa_SH_cms = Citys()
        self.accuracy_list = []

    def sh2Tst(self, city_name):
        if city_name == "qd":
            self.sh2_tsts.qd = SHH2ML_TrainImdc("qd")
            self.sh2_tsts.qd.csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_4_spl2.csv"
            self.sh2_tsts.qd.main()
            self.sh2_tsts.qd.train()
            self.sh2_tsts.qd.toToJson()
            fn = self.sh2_tsts.qd.td.time_dfn.fn()
            return os.path.split(fn)[1]
        elif city_name == "bj":
            self.sh2_tsts.bj = SHH2ML_TrainImdc("bj")
            self.sh2_tsts.bj.csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv"
            self.sh2_tsts.bj.main()
            self.sh2_tsts.bj.train()
            self.sh2_tsts.bj.toToJson()
            fn = self.sh2_tsts.bj.td.time_dfn.fn()
            return os.path.split(fn)[1]
        elif city_name == "cd":
            self.sh2_tsts.cd = SHH2ML_TrainImdc("cd")
            self.sh2_tsts.cd.csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\2\sh2_spl26_4_spl2.csv"
            self.sh2_tsts.cd.main()
            self.sh2_tsts.cd.train()
            self.sh2_tsts.cd.toToJson()
            fn = self.sh2_tsts.cd.td.time_dfn.fn()
            return os.path.split(fn)[1]

    def accuracyCM(self, accuracy_name, city_name, citys, map_dict1, map_dict2, cnames):
        self.td.kw("{} {}".format(accuracy_name, "MAP_DICT1"), map_dict1)
        self.td.kw("{} {}".format(accuracy_name, "MAP_DICT2"), map_dict2)
        self.td.kw("{} {}".format(accuracy_name, "CNAMES"), cnames)

        def _oa_kappa_cm(_json_fn):
            json_dict = FRW(_json_fn).readJson()
            to_dict = {}
            for k in json_dict:
                data = json_dict[k]
                y1 = mapDict(data["CNAME"], map_dict1)
                y2 = mapDict(data["y2"], map_dict2)
                if len(y1) != len(y2):
                    data_tmp = 1
                cm = ConfusionMatrix(class_names=cnames)
                cm.addData(y1, y2)
                to_dict[k] = cm
            return to_dict

        def _oa_kappa_files(dirname):
            to_list = []
            n = 1
            while True:
                fn = os.path.join(dirname, "categorys{}.json".format(n))
                if os.path.isfile(fn):
                    to_list.append(_oa_kappa_cm(fn))
                else:
                    return to_list
                n += 1

        if city_name == "qd":
            citys.qd = _oa_kappa_files(self.dfns.qd.fn())

        elif city_name == "bj":
            citys.bj = _oa_kappa_files(self.dfns.bj.fn())

        elif city_name == "cd":
            citys.cd = _oa_kappa_files(self.dfns.cd.fn())

    def accuracyOAKappa(self, city_name):
        map_dict1 = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        map_dict2 = {1: 1, 2: 2, 3: 3, 4: 4}
        cnames = ["IS", "VEG", "SOIL", "WAT"]
        citys = self.oa_kappa_cms
        accuracy_name = "ACCURACY OA KAPPA"
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracyOAKappaISO(self, city_name):
        map_dict1 = {
            "IS": 1, "VEG": 0, "SOIL": 2, "WAT": 0,
            "IS_SH": 0, "VEG_SH": 0, "SOIL_SH": 0, "WAT_SH": 0
        }
        map_dict2 = {1: 1, 2: 2, 3: 2, 4: 2}
        cnames = ["IS", "SOIL"]
        citys = self.oa_kappa_IO_cms
        accuracy_name = "ACCURACY OA KAPPA IS SOIL"
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracyOAKappaSH(self, city_name):
        map_dict1 = {
            "IS": 0, "VEG": 0, "SOIL": 0, "WAT": 0,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        map_dict2 = {1: 1, 2: 2, 3: 3, 4: 4}
        cnames = ["IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]
        accuracy_name = "ACCURACY OA KAPPA SHADOW"
        citys = self.oa_kappa_SH_cms
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracyOAKappaIS(self, city_name):
        map_dict1 = {
            "IS": 1, "VEG": 2, "SOIL": 2, "WAT": 2,
            "IS_SH": 2, "VEG_SH": 2, "SOIL_SH": 2, "WAT_SH": 2
        }
        map_dict2 = {1: 1, 2: 2, 3: 2, 4: 2}
        cnames = ["IS", "NOIS"]
        accuracy_name = "ACCURACY OA KAPPA IS"
        citys = self.oa_kappa_IS_cms
        self.accuracyCM(accuracy_name, city_name, citys, map_dict1, map_dict2, cnames)

    def accuracy(self):

        def calculate_city_cm(name, citys_cms, ):

            def calculate(cm_list):
                df_list = []
                for cm_dict in cm_list:
                    df_dict = {}
                    for k in cm_dict:
                        cm = cm_dict[k]
                        df_dict[k] = {"{}OA".format(name): cm.OA(), "{}Kappa".format(name): cm.getKappa(), }
                    df_list.append(df_dict)
                df = pd.DataFrame(df_list[0])
                for df_dict in df_list[1:]:
                    df += pd.DataFrame(df_dict)
                df /= len(df_list)
                return df

            def add_df(df, city_name):
                _to_dict = {"City": [city_name, city_name], "Accuracy": list(df.index), **(df.to_dict("list"))}
                to_list.append({k: _to_dict[k][0] for k in _to_dict})
                to_list.append({k: _to_dict[k][1] for k in _to_dict})

            qd_df = calculate(citys_cms.qd)
            add_df(qd_df, "QingDao")

            bj_df = calculate(citys_cms.bj)
            add_df(bj_df, "BeiJing")

            cd_df = calculate(citys_cms.cd)
            add_df(cd_df, "ChengDu")

        to_list = []
        calculate_city_cm("", self.oa_kappa_cms)
        calculate_city_cm("IS_", self.oa_kappa_IS_cms)
        calculate_city_cm("IO_", self.oa_kappa_IO_cms)
        calculate_city_cm("SH_", self.oa_kappa_SH_cms)
        return pd.DataFrame(to_list)

    def saveCM(self):
        sw = self.td.buildWriteText("cm.md")

        def calculate_city_cm(name, citys_cms, ):
            sw.write("# {}\n".format(name))

            def calculate(cm_list, city_name):
                sw.write("## {}\n".format(city_name))
                df_list = []
                for i, cm_dict in enumerate(cm_list):
                    df_dict = {}
                    sw.write("### {}\n".format(i + 1))
                    for k in cm_dict:
                        sw.write("#### {}\n".format(k))
                        cm = cm_dict[k]
                        df_dict[k] = cm.fmtCM()
                        sw.write(df_dict[k])
                    df_list.append(df_dict)
                return {city_name: df_list}

            return {name: {
                **calculate(citys_cms.qd, "QingDao"),
                **calculate(citys_cms.bj, "BeiJing"),
                **calculate(citys_cms.cd, "ChengDu"),
            }}

        to_dict = {
            **calculate_city_cm("oa_kappa_cms", self.oa_kappa_cms),
            **calculate_city_cm("oa_kappa_IS_cms", self.oa_kappa_IS_cms),
            **calculate_city_cm("oa_kappa_IO_cms", self.oa_kappa_IO_cms),
            **calculate_city_cm("oa_kappa_SH_cms", self.oa_kappa_SH_cms),
        }

        self.td.saveJson("cm.json", to_dict)

    def main(self):
        self.dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods")
        self.td = TimeDirectory(self.dfn.fn())
        self.td.initLog()
        self.td.log("#", "-" * 34, "SHH2ML Three Accuracy", "-" * 34, "#")
        self.td.log(self.td.time_dfn.dirname)
        time.sleep(2)

        # QingDao
        self.td.log("#", "-" * 30, "SHH2ML Three Accuracy QingDao", "-" * 30, "#")
        # dirname = self.td.kw("QingDao Dirname", self.dfn.fn(self.sh2Tst("qd")))
        dirname = self.td.kw("QingDao Dirname", self.dfn.fn("20240620H100549"))
        self.dfns.qd = DirFileName(dirname)
        self.accuracyOAKappa("qd")
        self.accuracyOAKappaIS("qd")
        self.accuracyOAKappaSH("qd")
        self.accuracyOAKappaISO("qd")

        # BeiJing
        self.td.log("#", "-" * 30, "SHH2ML Three Accuracy BeiJing", "-" * 30, "#")
        # dirname = self.td.kw("BeiJing Dirname", self.dfn.fn(self.sh2Tst("bj")))
        dirname = self.td.kw("BeiJing Dirname", self.dfn.fn("20240620H100627"))
        self.dfns.bj = DirFileName(dirname)
        self.accuracyOAKappa("bj")
        self.accuracyOAKappaIS("bj")
        self.accuracyOAKappaSH("bj")
        self.accuracyOAKappaISO("bj")

        # ChengDu
        self.td.log("#", "-" * 30, "SHH2ML Three Accuracy ChengDu", "-" * 30, "#")
        # dirname = self.td.kw("ChengDu Dirname", self.dfn.fn(self.sh2Tst("cd")))
        dirname = self.td.kw("ChengDu Dirname", self.dfn.fn("20240704H113806"))
        self.dfns.cd = DirFileName(dirname)
        self.accuracyOAKappa("cd")
        self.accuracyOAKappaIS("cd")
        self.accuracyOAKappaSH("cd")
        self.accuracyOAKappaISO("cd")

        df = self.accuracy()
        print(self.td.saveDF("accuracy.csv", df))
        print(df)
        self.saveCM()
        return


def shh2ml_tst_three_main():
    def func1():
        sh2_tst_three = SHH2ML_TST_Three_Main()
        sh2_tst_three.main()

    def func2():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240704H121321\accuracy.csv")

        class df_filter:

            def __init__(self, _df):
                self.df = _df

            def eq(self, field_name, data):
                if not isinstance(data, list):
                    data = [data]
                self.df = self.df[self.df[field_name].isin(data)]
                return self

            def datain(self, field_name, data):
                self.df = self.df[self.df[field_name].apply(lambda x: data in x)]
                return self

        city_name = "ChengDu"
        df = df_filter(df) \
            .datain("Accuracy", "OA") \
            .df.reset_index(drop=True)
        #             .eq("City", city_name) \

        print(df, "\n")
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        df2 = df[fns]
        df2 = df2.reset_index(drop=True)
        df2.index = ["{} {}".format(df["City"][i], df["Accuracy"][i], ) for i in range(len(df))]
        print(df2, "\n")
        print(df2.T.sort_values("{} IS_OA".format(city_name), ascending=False), "\n")

    func2()


def main():
    def bj():
        sh2_at = SHH2AccTiao(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\sh2_spl27_21_spl.csv")
        sh2_at.addQJY(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\sh2_spl27_3_test_spl.txt")
        sh2_at.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\sh2_spl27_3_test_spl.csv")
        sh2_at.y1_map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        sh2_at.y2_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
        sh2_at.cnames = ["IS", "VEG", "SOIL", "WAT", ]
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        sh2_at.addGSICDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240609H091804", fns)
        to_dict_list = sh2_at.tacc1()
        sh2_at.saveAccuracy(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\1", "bjtest", acc=to_dict_list)

    def cd():
        sh2_at = SHH2AccTiao(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\shh2_spl26_4_random800_spl2.csv")
        sh2_at.addQJY(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\shh2_spl26_5_test.txt")
        sh2_at.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\shh2_spl26_4_random800_spl2_at.csv")
        sh2_at.y1_map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
        }
        sh2_at.y2_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
        sh2_at.cnames = ["IS", "VEG", "SOIL", "WAT", ]
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        sh2_at.addGSICDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240609H091808", fns)
        to_dict_list = sh2_at.tacc1()
        sh2_at.saveAccuracy(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\1", "bjtest", acc=to_dict_list)

    def qd():
        sh2_at = SHH2AccTiao(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_5_test.csv")
        # sh2_at.addQJY(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_5_test.txt")
        sh2_at.toCSV(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_5_test_at.csv")
        sh2_at.y1_map_dict = {
            "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
            "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4,
        }
        sh2_at.y2_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
        sh2_at.cnames = ["IS", "VEG", "SOIL", "WAT", ]
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
        sh2_at.addGSICDirectory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240616H204930", fns)
        to_dict_list = sh2_at.tacc1([["IS_SH"], ["VEG_SH", "SOIL_SH", "SOIL_SH", "WAT_SH"]])
        sh2_at.saveAccuracy(r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\1", "qdtest", acc=to_dict_list)

    qd()
    return


def method_name2():
    # filelist = DFN(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240608H171757").filterFileEndWith("_imdc.tif", False)
    # filelist = list(map(lambda x: x.replace("_imdc.tif", ""), filelist))
    # filelist.sort()
    # print(filelist)
    # csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\shh2_spl26_4_random800_spl2.csv" # ChengDu
    # csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_4_test.csv" # QingDao
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\sh2_spl27_21_spl.csv"  # BeiJing
    to_csn_fn = FN(csv_fn).changext("_acc.csv")
    df = pd.read_csv(csv_fn)
    # df = df[df["TEST"] == 0]
    x, y = df["X"].tolist(), df["Y"].tolist()
    y1 = df["CNAME"].tolist()
    y1_map_dict = {
        "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
        "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
    }
    y2_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
    cnames = ["IS", "VEG", "SOIL", "WAT", ]
    gsic = GDALSamplingImageClassification()
    # fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM', 'OPT+BS+C2+SARGLCM', ]
    fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']

    def gsic_add_directory(dirname):
        for fn in fns:
            gsic.add(fn, os.path.join(dirname, fn + "_imdc.tif"))
            print(os.path.join(dirname, fn + "_imdc.tif"))

    # gsic_add_directory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240609H091757") # ChengDu
    # gsic_add_directory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240609H091808") # QingDao
    gsic_add_directory(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240609H091804")  # BeiJing
    # gsic.add("DL_OPT", r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H092654\OPT_epoch36_imdc1.tif")
    # gsic.add("DL_OPTASDE", r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H100838\OPTASDE_epoch36_imdc1.tif")
    # gsic.add("FC_SVM", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240524H090841fc\qd_sh2_1_opt_sar_glcm_fc_imdc.tif")
    to_list = []

    def acc(y2_name):
        print(y2_name)
        sa = SRTAccuracy()
        sa.y1 = y1
        sa.y1_map_dict = y1_map_dict
        sa.y2 = gsic[y2_name].sampling(x, y)
        sa.y2_map_dict = y2_map_dict
        sa.cnames = cnames
        cm = sa.cm()
        print(cm.fmtCM())
        is_cm = cm.accuracyCategory("IS")
        print(is_cm.fmtCM())
        to_list.append({"NAME": y2_name, "IS_OA": is_cm.OA(), "IS_Kappa": is_cm.getKappa(), "OA": cm.OA(),
                        "Kappa": cm.getKappa()})

    for k in gsic.keys():
        acc(k)
    print(pd.DataFrame(to_list).sort_values("IS_OA", ascending=False))

    def y_tf():
        sort_list = ["Y_SUM"]
        to_dict = gsic.fit(x, y, to_dict=df.to_dict("list"), map_dict=y2_map_dict)

        # y_sum = None
        # for k in gsic.keys():
        #     to_dict["{}_{}".format(k, "TF")] = ((np.array(to_dict[k]) == np.array(to_dict["CATEGORY"])) * 1).tolist()
        #     if y_sum is None:
        #         y_sum = np.array(to_dict["{}_{}".format(k, "TF")])
        #     else:
        #         y_sum += np.array(to_dict["{}_{}".format(k, "TF")])
        #     sort_list.append(k)
        # to_dict["Y_SUM"] = y_sum.tolist()

        pd.DataFrame(to_dict).to_csv(to_csn_fn, index=False)
    # y_tf()


def method_name1():
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_2_spl2_2.csv"
    csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_soil1.csv"
    df = pd.read_csv(csv_fn)
    # df = df[df["TEST"] == 0]
    x, y = df["X"].tolist(), df["Y"].tolist()
    y1 = df["CNAME"].tolist()
    y1_map_dict = {
        "IS": 1, "VEG": 2, "SOIL": 3, "WAT": 4,
        "IS_SH": 1, "VEG_SH": 2, "SOIL_SH": 3, "WAT_SH": 4
    }
    y2_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
    cnames = ["IS", "VEG", "SOIL", "WAT", ]
    gsic = GDALSamplingImageClassification()
    gsic.add("DL_OPT", r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H092654\OPT_epoch36_imdc1.tif")
    gsic.add("DL_OPTASDE", r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240607H100838\OPTASDE_epoch36_imdc1.tif")
    gsic.add("OPT", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT_imdc.tif")
    gsic.add("OPT+AS", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+AS_imdc.tif")
    gsic.add("OPT+AS+DE", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+AS+DE_imdc.tif")
    gsic.add("OPT+BS", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+BS_imdc.tif")
    gsic.add("OPT+BS+C2+SARGLCM",
             r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+BS+C2+SARGLCM_imdc.tif")
    gsic.add("OPT+C2", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+C2_imdc.tif")
    gsic.add("OPT+DE", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+DE_imdc.tif")
    gsic.add("OPT+HA", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+HA_imdc.tif")
    gsic.add("OPT+SARGLCM", r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240606H100828\OPT+SARGLCM_imdc.tif")
    gsic.add("FC_SVM", r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240524H090841fc\qd_sh2_1_opt_sar_glcm_fc_imdc.tif")
    to_list = []

    def acc(y2_name):
        print(y2_name)
        sa = SRTAccuracy()
        sa.y1 = y1
        sa.y1_map_dict = y1_map_dict
        sa.y2 = gsic[y2_name].sampling(x, y)
        sa.y2_map_dict = y2_map_dict
        sa.cnames = cnames
        cm = sa.cm()
        print(cm.fmtCM())
        is_cm = cm.accuracyCategory("IS")
        print(is_cm.fmtCM())
        to_list.append({"NAME": y2_name, "IS_OA": is_cm.OA(), "IS_Kappa": is_cm.getKappa(), "OA": is_cm.OA(),
                        "Kappa": is_cm.getKappa()})

    # for k in gsic.keys():
    #     acc(k)
    # print(pd.DataFrame(to_list).sort_values("IS_OA", ascending=False))
    to_dict = gsic.fit(x, y, to_dict=df.to_dict("list"), map_dict=y2_map_dict)

    def y_tf():
        y_sum = None
        for k in gsic.keys():
            to_dict["{}_{}".format(k, "TF")] = ((np.array(to_dict[k]) == np.array(to_dict["CATEGORY"])) * 1).tolist()
            if y_sum is None:
                y_sum = np.array(to_dict["{}_{}".format(k, "TF")])
            else:
                y_sum += np.array(to_dict["{}_{}".format(k, "TF")])
            sort_list.append(k)
        to_dict["Y_SUM"] = y_sum.tolist()

    sort_list = ["Y_SUM"]
    y_tf()
    pd.DataFrame(to_dict) \
        .sort_values(sort_list) \
        .to_csv(
        r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\sh2_spl25_soil2.csv", index=False)


if __name__ == "__main__":
    main()
