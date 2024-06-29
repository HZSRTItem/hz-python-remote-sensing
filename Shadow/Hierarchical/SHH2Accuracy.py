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

import numpy as np
import pandas as pd

from SRTCodes.GDALUtils import GDALSamplingImageClassification, GDALSamplingFast, GDALSampling
from SRTCodes.TrainingUtils import SRTAccuracy
from SRTCodes.Utils import FN, TimeName, FRW
from Shadow.ShadowUtils import readQJYTxt


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


class SHH2SamplesManage:

    def __init__(self):
        self.spl_fns = []
        self.df = None
        self.x_list = []
        self.y_list = []
        self.c_list = []
        self.x_field_name = "X"
        self.y_field_name = "Y"
        self.c_field_name = "CNAME"

    def addDF(self, df, fun_df=None, field_datas=None):
        if field_datas is None:
            field_datas = {}
        df = df.copy()
        for k in field_datas:
            df[k] = [field_datas[k] for _ in range(len(df))]
        if fun_df is not None:
            df = fun_df(df)
        if self.df is None:
            self.df = df
        else:
            df_temp = self.df.to_dict("records")
            df_temp.extend(df.to_dict("records"))
            self.df = pd.DataFrame(df_temp)

        self.x_list.extend(df[self.x_field_name].tolist())
        self.y_list.extend(df[self.y_field_name].tolist())
        self.c_list.extend(df[self.c_field_name].tolist())

        return df

    def addCSVS(self, *csv_fns, fun_df=None, field_datas=None):
        for csv_fn in csv_fns:
            df = pd.read_csv(csv_fn)
            self.addDF(df, fun_df=fun_df, field_datas=field_datas)
            self.spl_fns.append(csv_fn)

    def addQJY(self, txt_fn, fun_df=None, field_datas=None):
        df_dict = readQJYTxt(txt_fn)
        x = df_dict["__X"]
        y = df_dict["__Y"]
        c_name = df_dict["__CNAME"]
        df_dict[self.x_field_name] = x
        df_dict[self.y_field_name] = y
        df_dict[self.c_field_name] = c_name
        df = pd.DataFrame(df_dict)
        df = self.addDF(df, fun_df=fun_df, field_datas=field_datas)
        self.spl_fns.append(txt_fn)
        return df

    def toDF(self, x_field_name=None, y_field_name=None, c_field_name=None) -> pd.DataFrame:
        if x_field_name is None:
            x_field_name = self.x_field_name
        if y_field_name is None:
            y_field_name = self.y_field_name
        if c_field_name is None:
            c_field_name = self.c_field_name
        df = self.df.copy()
        df[x_field_name] = self.x_list
        df[y_field_name] = self.y_list
        df[c_field_name] = self.c_list
        return df

    def toCSV(self, csv_fn, x_field_name=None, y_field_name=None, c_field_name=None):
        self.toDF(x_field_name, y_field_name, c_field_name).to_csv(csv_fn, index=False)

    def __len__(self):
        return len(self.c_list)

    def sampling(self, raster_fn, spl_type="fast", x_field_name=None, y_field_name=None, c_field_name=None):
        if spl_type == "fast":
            gs = GDALSamplingFast(raster_fn)
        elif spl_type == "iter":
            gs = GDALSampling(raster_fn)
        elif spl_type == "npy":
            gs = GDALSampling()
            gs.initNPYRaster(raster_fn)
        else:
            raise Exception("Can not format sampling type of \"{}\"".format(spl_type))
        to_df = self.toDF(x_field_name, y_field_name, c_field_name)
        to_df = gs.samplingDF(to_df)
        return to_df


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

        cm = accuracyY12(y1, y2, y1_map_dict, y2_map_dict, cnames, fc_category,is_eq_number=is_eq_number)

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
