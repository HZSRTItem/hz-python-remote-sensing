# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Sample.py
@Time    : 2024/6/8 16:56
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Sample
-----------------------------------------------------------------------------"""
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from tabulate import tabulate

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import GDALSampling, GDALSamplingFast, GDALNumpySampling, uniqueSamples
from SRTCodes.OGRUtils import sampleSpaceUniform
from SRTCodes.SRTData import DFF_AND
from SRTCodes.SRTLinux import W2LF
from SRTCodes.SRTModel import mapDict
from SRTCodes.SRTSample import GEOJsonWriteWGS84, dfnumber, readQJYTxt
from SRTCodes.Utils import getRandom, getfilenamewithoutext, DirFileName, FN, Jdt, saveJson
from Shadow.Hierarchical import SHH2Config, SHHConfig

RESOLUTION_ANGLE = 0.000089831529294


def _GS_NPY(npy_fn):
    gs = GDALSampling()
    gs.initNPYRaster(npy_fn)
    return gs


def QD_GS_NPY():
    return _GS_NPY(SHH2Config.QD_NPY_FN)


def BJ_GS_NPY():
    return _GS_NPY(SHH2Config.BJ_NPY_FN)


def CD_GS_NPY():
    return _GS_NPY(SHH2Config.CD_NPY_FN)


def SAMPLING_CITY_NAME(city_name, csv_fn, to_csv_fn=None):
    if to_csv_fn is None:
        to_csv_fn = csv_fn
    GDALSamplingFast(SHH2Config.GET_RASTER_FN(city_name)).csvfile(csv_fn=csv_fn, to_csv_fn=to_csv_fn)


def shh2ShowCSVNumbers(*csv_fns, name1, name2, is_print=True):
    to_list = []
    for csv_fn in csv_fns:
        df_list = pd.read_csv(csv_fn).to_dict("records")
        to_list.extend([{name1: spl[name1], name2: spl[name2]} for spl in df_list])
    return dfnumber(pd.DataFrame(to_list), name1, name2, is_print=is_print)


class _SplNum:

    def __init__(self, name, **category_numbers):
        self.name = name
        self.category_numbers = category_numbers
        self.datas = {name: [] for name in category_numbers}

    def __setitem__(self, key, value):
        self.category_numbers[key] = value
        self.datas[key] = []

    def __getitem__(self, item):
        return self.category_numbers[item]

    def __contains__(self, item):
        return item in self.category_numbers

    def __len__(self):
        return len(self.category_numbers)

    def endCnames(self):
        return list(self.category_numbers.keys())

    def getNumber(self, end_cnames, is_curr=False):
        to_list = []
        for name in end_cnames:
            if name in self.category_numbers:
                if not is_curr:
                    to_list.append(self.category_numbers[name])
                else:
                    to_list.append(len(self.datas[name]))
            else:
                to_list.append(0)
        return to_list

    def add(self, samples, select_list, fc_name, sub_name):
        for i, spl in enumerate(samples):
            if select_list[i]:
                for name in self.category_numbers:
                    if (spl["CNAME"] == name) and (self.category_numbers[name] > len(self.datas[name])):
                        spl["FCNAME"] = fc_name
                        spl["SUBNAME"] = sub_name
                        self.datas[name].append(spl)
                        select_list[i] = False
        return select_list

    def getName(self, n):
        return list(self.category_numbers.keys())[n]

    def getSamples(self):
        to_list = []
        for name in self.datas:
            to_list.extend(self.datas[name])
        return to_list


class SHH2SamplesNumbers:

    def __init__(self, end_cnames=None):
        self.numbers = {}
        self.end_cnames = end_cnames
        self.samples = []

    def add(self, cname, sub_cname, **numbers):
        if cname not in self.numbers:
            self.numbers[cname] = {}
        self.numbers[cname][sub_cname] = _SplNum(sub_cname, **numbers)

    def __getitem__(self, item):
        return self.numbers[item]

    def show(self, _type="numbers"):
        if self.end_cnames is None:
            self.endCnames()

        def get_numbers(is_curr=False):
            data = {}
            n_all_sum = None
            for _name in self.numbers:
                data[_name] = {}
                n_sum_list = []
                n = 1
                for _sub_name in self.numbers[_name]:
                    n_list = self.numbers[_name][_sub_name].getNumber(self.end_cnames, is_curr=is_curr)
                    data[_name][_sub_name] = n_list
                    if n == 1:
                        n_sum_list = n_list
                    else:
                        n_sum_list = [n_list[i] + n_sum_list[i] for i in range(len(n_sum_list))]
                    if n_all_sum is None:
                        n_all_sum = n_list
                    else:
                        n_all_sum = [n_list[i] + n_all_sum[i] for i in range(len(n_all_sum))]
                    n += 1
                data[_name]["SUM"] = n_sum_list
            data["SUM"] = {" ": n_all_sum}
            return data

        if _type == "numbers":
            headers = ["GROUPS", "CNAME"] + self.end_cnames + ["SUM"]
            lines = get_numbers()
            to_data = []
            for name in self.numbers:
                for n, sub_name in enumerate(self.numbers[name]):
                    to_data.append([name if n == 0 else " ", sub_name] + lines[name][sub_name])
                to_data.append([" ", "SUM"] + lines[name]["SUM"])
            to_data.append(["SUM", " "] + lines["SUM"][" "])
            print(tabulate(tabular_data=to_data, headers=headers, tablefmt="simple"))

        elif _type == "current":
            headers = ["GROUPS", "SUB-CNAME"] + self.end_cnames + ["SUM"]
            lines = get_numbers()
            lines_curr = get_numbers(True)
            to_add = []

            def get_data(_name, _sub_name):
                n1 = lines[_name][_sub_name]
                n2 = lines_curr[_name][_sub_name]
                _to_list = []
                for i in range(len(n1)):
                    if not ((n1[i] == 0) and (n2[i] == 0)):
                        _to_list.append("{}->{}".format(n2[i], n1[i]))
                        if (n2[i] < n1[i]) and (_name != "SUM") and (_sub_name != "SUM"):
                            to_add.append((_name, _sub_name, self.end_cnames[i], n2[i], n1[i], n1[i] - n2[i]))
                    else:
                        _to_list.append("0")
                _to_list.append("{}->{}".format(sum(n2), sum(n1)))
                return _to_list

            to_data = []
            for name in self.numbers:
                for n, sub_name in enumerate(self.numbers[name]):
                    to_data.append([name if n == 0 else " ", sub_name] + get_data(name, sub_name))
                to_data.append(["------"] * len(headers))
                to_data.append([" ", "SUM"] + get_data(name, "SUM"))
                to_data.append(["******"] * len(headers))
            to_data.append(["SUM", " "] + get_data("SUM", " "))
            print("# NUMBERS ------")
            print(tabulate(tabular_data=to_data, headers=headers, tablefmt="simple"))
            print("# ADD NUMBERS ------")
            print(tabulate(
                tabular_data=to_add, headers=["GROUPS", "SUB-CNAME", "CNAME", "N", "TO N", "dN"],
                tablefmt="simple"
            ))

        else:
            raise Exception("Can not format _type as {}".format(_type))

    def endCnames(self):
        to_list = []
        for name in self.numbers:
            for _, spl in self.numbers[name].items():
                to_list.extend(spl.endCnames())
        to_list = list(set(to_list))
        self.end_cnames = to_list

    def addSamples(self, samples, field_datas=None):
        if field_datas is None:
            field_datas = {}
        for spl in samples:
            for name, data in field_datas.items():
                spl[name] = data
        select_list = [True for _ in range(len(samples))]
        random.shuffle(samples)
        for name, sub_dict in self.numbers.items():
            for sub_name, spln in sub_dict.items():
                spln.add(samples, select_list, name, sub_name)
        self.samples.extend(samples)

    def addSamplesCSV(self, csv_fn, field_datas=None):
        if field_datas is None:
            field_datas = {}
        field_datas["SOURCE"] = csv_fn
        return self.addSamples(pd.read_csv(csv_fn).to_dict("records"), field_datas=field_datas)

    def addSamplesQJY(self, txt_fn, field_datas=None):
        # df_dict["__X"][-1] = x
        # df_dict["__Y"][-1] = y
        # df_dict["__CNAME"][-1] = c_name
        # df_dict["__IS_TAG"][-1] = is_tag
        df_dict = readQJYTxt(txt_fn)
        if "OSRT" not in df_dict:
            df_dict["OSRT"] = [i + 1 for i in range(len(df_dict["__X"]))]
        df = pd.DataFrame(df_dict)
        df = df.rename(columns={"__X": "X", "__Y": "Y", "__CNAME": "CNAME"})
        if field_datas is None:
            field_datas = {}
        field_datas["SOURCE"] = txt_fn
        field_datas["LOOK"] = 1
        return self.addSamples(df.to_dict("records"), field_datas)

    def scatter(self, c_field_name="CNAME", color_dict=None):
        if color_dict is None:
            color_dict = {}
        samples = self.getSamples()
        x, y, cnames = [], [], []
        for spl in samples:
            x.append(spl["X"])
            y.append(spl["Y"])
            cnames.append(spl[c_field_name])
        df = pd.DataFrame({"X": x, "Y": y, "CNAME": cnames})
        cnames_show = pd.unique(df["CNAME"])
        for cname in cnames_show:
            _df = df[df["CNAME"] == cname]
            plt.scatter(_df["X"], _df["Y"], label=cname, color=None if cname not in color_dict else color_dict[cname],
                        s=2)
        plt.gca().set_aspect(1)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

    def getSamples(self):
        samples = []
        for name in self.numbers:
            for n, (sub_name, spln) in enumerate(self.numbers[name].items()):
                samples.extend(spln.getSamples())
        return samples

    def toDF(self, samples=None):
        if samples is None:
            samples = []
        return pd.DataFrame(self.getSamples() + samples)

    def toCSV(self, csv_fn, select_names=None, samples=None):
        if samples is None:
            samples = []
        df = self.toDF(samples=samples)
        if "SRT" in df:
            if "OSRT" in df:
                df = df.rename(columns={"OSRT": "OSRT2"})
            df = df.rename(columns={"SRT": "OSRT"})
        df.index += 1
        if select_names is not None:
            df = df[select_names]
        df.to_csv(csv_fn, index_label="SRT")

        print("# NUMBERS ALL CSV FILES ------")
        df1 = dfnumber(df, "CNAME", "TEST", is_print=False).rename(columns={0: "To 0", 1: "To 1", "SUM": "To SUM"})
        df2 = dfnumber(pd.DataFrame(self.samples), "CNAME", "TEST", is_print=False).rename(
            columns={0: "Original 0", 1: "Original 1", "SUM": "Original SUM"})
        show_df = pd.concat([df1, df2], axis=1)
        try:
            show_df["More 0"] = show_df["Original 0"] - show_df["To 0"]
        except Exception as e:
            print(e)
        try:
            show_df["More 1"] = show_df["Original 1"] - show_df["To 1"]
        except Exception as e:
            print(e)
        show_df["More SUM"] = show_df["Original SUM"] - show_df["To SUM"]
        print(tabulate(show_df, "keys"))

        return df


def sampling():
    def qd():
        GDALSamplingFast(SHH2Config.QD_ENVI_FN).csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd2_random6000.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd2_random6000_spl.csv",
        )

    def bj():
        GDALSamplingFast(SHH2Config.BJ_ENVI_FN).csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj1.csv",
        )

    def cd():
        GDALSamplingFast(SHH2Config.CD_ENVI_FN).csvfile(
            csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6.csv",
            to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv",
        )

    # qd()
    # bj()
    cd()


class SHH2Sampling:

    def __init__(self, csv_fn):
        self.csv_fn = None
        self.dirname1 = r"F:\ProjectSet\Shadow\Hierarchical\Samples\ML"
        self.dirname2 = r"F:\ProjectSet\Shadow\Hierarchical\Samples\DL"

    def get(self):

        return

    def get2(self, win_rows, win_columns):
        csv_fn = self.csv_fn

        def getFileName():
            dfn = DirFileName(self.dirname2)
            _to_fn = dfn.fn("{}-{}_{}.csv".format(FN(csv_fn).getfilenamewithoutext(), win_rows, win_columns))
            _to_npy_fn = FN(to_fn).changext("-data.npy")
            return to_fn, to_npy_fn

        class sample:

            def __init__(self, _line):
                self.x = _line["X"]
                self.y = _line["Y"]
                self.city = None

        to_fn, to_npy_fn = getFileName()

        gr = {"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }

        n_channels = gr["qd"].n_channels

        df = pd.read_csv(csv_fn)
        samples = []
        numbers = {"qd": 0, "bj": 0, "cd": 0}
        data = np.zeros((len(df), n_channels, win_rows, win_columns))
        data_shape = data[0].shape

        jdt = Jdt(len(df), "SHH2Sampling Sampling").start()
        for i in range(len(df)):
            line = df.loc[i]
            spl = sample(line)
            for k in gr:
                if gr[k].isGeoIn(spl.x, spl.y):
                    spl.city = k
                    break
            if spl.city is None:
                warnings.warn("{}, {} not in raster.".format(spl.x, spl.y))
                jdt.add()
            else:
                numbers[spl.city] += 1
            samples.append(spl)
        print(numbers)
        for k in numbers:
            if numbers[k] != 0:
                gr[k].readAsArray()
                # gr[k].d = np.zeros((3, gr[k].n_rows, gr[k].n_columns))
                gns = GDALNumpySampling(win_rows, win_columns, gr[k])
                for i, spl in enumerate(samples):
                    data_i = gns.getxy(spl.x, spl.y)
                    if data_shape == data_i.shape:
                        data[i] = gns.getxy(spl.x, spl.y)
                    else:
                        spl.city = None
                        warnings.warn("{}, {} not in raster.".format(spl.x, spl.y))
                    jdt.add()
                gns.data = None
                gr[k].d = None

        jdt.end()
        df["city"] = [str(spl.city) for spl in samples]
        df.to_csv(to_fn, index=False)
        print(to_fn)
        np.save(to_npy_fn, data.astype("float16"))

        return


def samplingCSVData(csv_fn, to_fn, to_npy_fn, names_fn, win_rows, win_columns):
    gr = {"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }
    n_channels = gr["qd"].n_channels
    df = pd.read_csv(csv_fn)
    samples = []
    numbers = {"qd": 0, "bj": 0, "cd": 0}
    data = np.zeros((len(df), n_channels, win_rows, win_columns), dtype="float32")
    data_shape = data[0].shape

    class sample:

        def __init__(self, _line):
            self.x = _line["X"]
            self.y = _line["Y"]
            self.city = None

    jdt = Jdt(len(df), "SHH2DL Sampling").start()
    for i in range(len(df)):
        line = df.loc[i]
        spl = sample(line)
        for k in gr:
            if gr[k].isGeoIn(spl.x, spl.y):
                spl.city = k
                break
        if spl.city is None:
            warnings.warn("{}, {} not in raster.".format(spl.x, spl.y))
            jdt.add()
        else:
            numbers[spl.city] += 1
        samples.append(spl)
    for k in numbers:
        if numbers[k] != 0:
            gr[k].readAsArray()
            gns = GDALNumpySampling(win_rows, win_columns, gr[k])
            for i, spl in enumerate(samples):
                data_i = gns.getxy(spl.x, spl.y)
                if data_shape == data_i.shape:
                    data[i] = gns.getxy(spl.x, spl.y)
                else:
                    spl.city = None
                    warnings.warn("{}, {} not in raster.".format(spl.x, spl.y))
                jdt.add()
            gns.data = None
            gr[k].d = None
    jdt.end()
    df["city"] = [str(spl.city) for spl in samples]
    df.to_csv(to_fn, index=False)
    print(to_fn)
    np.save(to_npy_fn, data)

    saveJson(gr["qd"].names, names_fn)
    return gr["qd"].names


def samplingTest():
    gs = GDALSamplingFast(SHH2Config.QD_LOOK_FN)
    gs.csvfile(
        csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_test_random1000.csv",
        to_csv_fn=r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\2\sh2_spl252_test_random1000_spl.csv",
    )


def randomSamples(n, x0, x1, y0, y1):
    return {
        "SRT": [i + 1 for i in range(n)],
        "X": [getRandom(x0, x1) for _ in range(n)],
        "Y": [getRandom(y0, y1) for _ in range(n)],
    }


def samplingImdc(df, dirname, fns=None):
    if fns is None:
        fns = ['OPT', 'OPT+AS', 'OPT+DE', 'OPT+AS+DE', 'OPT+GLCM', 'OPT+BS', 'OPT+C2', 'OPT+HA', 'OPT+SARGLCM']
    images = {fn: os.path.join(dirname, fn + "_imdc.tif") for fn in fns}
    gss = {fn: GDALSampling(os.path.join(dirname, fn + "_imdc.tif")) for fn in fns}
    for fn in images:
        gs = gss[fn]
        categorys = gs.sampling(df["X"].tolist(), df["Y"].tolist())
        for name in categorys:
            df[fn] = categorys[name]
            break
    return fns


def main():
    def func1():
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\29")
        raster_fn = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\20240626H135521\Three_epoch88_imdc1.tif"
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl.csv"
        to_fn = dfn.fn("sh2_spl29_qd1.csv")
        df = pd.read_csv(csv_fn)
        df = GDALSampling(raster_fn).samplingDF(df)

        map_dict = {}
        for i in range(len(df)):
            line = df.loc[i]
            data1 = df["FEATURE_1"]

        df.to_csv(to_fn, index=False)

    return func1()


def method_name1():
    def func1():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl27_1.csv")
        coors2, out_index_list = sampleSpaceUniform(df[["X", "Y"]].values.tolist(), x_len=1200, y_len=1200,
                                                    is_trans_jiaodu=True, ret_index=True)
        is_save = np.zeros(len(df))
        is_save[out_index_list] = 1
        df["IS_SAVE"] = is_save
        df = df[df["IS_SAVE"] == 1]
        print(len(out_index_list))
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl27_1_ssu.csv", index=False)

        return

    def func2():
        # beijing 116.265589506, 116.766938820, 39.614856934, 39.966812062
        # chengdu 103.766340133, 104.184577287, 30.537178820, 30.842317360
        # qingdao 119.974220271, 120.481110190, 36.044704313, 36.403399024
        df = pd.DataFrame(randomSamples(1000, 116.265589506, 116.766938820, 39.614856934, 39.966812062))
        print(df)
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_1.csv", index=False)

    def func3():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_1.csv")
        to_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\2\sh2_spl271_2.csv"
        dirname = r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240609H091804"

        fns = samplingImdc(df, dirname)

        print(df.keys())
        data = df[fns].values
        category = []
        for i in range(len(df)):
            if len(np.unique(data[i])) == 1:
                category.append(data[i, 0])
            else:
                category.append(0)
        df["CATEGORY"] = category

        df.to_csv(to_csv_fn, index=False)
        BJ_GS_NPY().csvfile(csv_fn=to_csv_fn, to_csv_fn=to_csv_fn, )
        print(df)

    def func4():
        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\spl1.csv")

        def scatter_name(c_field_name, cname, color):
            _df = df[df[c_field_name] == cname]
            plt.scatter(_df[x_field_name], _df[y_field_name], label=cname, color=color)

        x_field_name, y_field_name = "NDVI", "NDWI"
        scatter_name("CNAME", "VEG", "green")
        # scatter_name("CNAME", "WAT", "blue")
        scatter_name("CNAME", "SOIL", "yellow")
        # scatter_name("CNAME", "IS", "red")

        plt.legend()
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xlabel(x_field_name)
        plt.ylabel(y_field_name)
        plt.show()

    def func5():
        gr = GDALRaster(SHH2Config.CD_ENVI_FN)
        data = gr.readGDALBand("NDWI")
        gr.save(data, r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\data\ndwi.dat")

    def func6():
        fns = [
            r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+AS_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+AS+DE_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+BS_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+C2_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+DE_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+GLCM_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+HA_imdc.tif"
            , r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240614H170915\OPT+SARGLCM_imdc.tif"
        ]

        df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist2.csv")

        for fn in fns:
            gs = GDALSamplingFast(fn)
            data = gs.sampling(df["X"].tolist(), df["Y"].tolist())
            name = getfilenamewithoutext(fn)
            df[name] = data["FEATURE_1"]
        df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist3.csv", index=False)

    def func7():
        x_len, y_len = RESOLUTION_ANGLE * 200, RESOLUTION_ANGLE * 200
        gr = GDALRaster(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\SHH2_CD2_look.tif")
        x0, x1, y0, y1 = gr.raster_range
        n_x = int((x1 - x0) / x_len) + 1
        n_y = int((y1 - y0) / y_len) + 1
        gjw = GEOJsonWriteWGS84("SRT")
        n = 1
        to_dict = {"X": [], "Y": [], "SRT": [], "CATEGORY": []}
        for i in range(n_x):
            for j in range(n_y):
                gjw.addPolygon([[
                    [x0 + i * x_len, y0 + j * y_len, ],
                    [x0 + i * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + j * y_len, ],
                    [x0 + i * x_len, y0 + j * y_len, ],
                ]], SRT=n)
                data = np.array([
                    [x0 + i * x_len, y0 + j * y_len, ],
                    [x0 + i * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + (j + 1) * y_len, ],
                    [x0 + (i + 1) * x_len, y0 + j * y_len, ],
                ]).mean(axis=0)
                to_dict["X"].append(float(data[0]))
                to_dict["Y"].append(float(data[1]))
                to_dict["SRT"].append(int(n))
                to_dict["CATEGORY"].append(1)
                n += 1

        gjw.save(r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist_grids1.geojson")
        print(pd.DataFrame(to_dict))
        pd.DataFrame(to_dict).to_csv(
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\26\is_soil_test\sh2_spl26_ist_grids1.csv")

    func1()


def funcs():
    return


def _COM(data1, f, data2):
    if f == "==":
        return data1 == data2
    if f == ">":
        return data1 > data2
    if f == "<":
        return data1 < data2
    if f == ">=":
        return data1 >= data2
    if f == "<=":
        return data1 <= data2
    raise Exception("{}".format(f))


def samplesFilterOR(samples, *filters):
    to_list = []
    for spl in samples:
        for name, f, data in filters:
            if _COM(spl[name], f, data):
                to_list.append(spl)
                break
    return to_list


def samplesFind():
    region_dict = {
        "beijing": [116.265589506, 116.766938820, 39.614856934, 39.966812062],
        "chengdu": [103.766340133, 104.184577287, 30.537178820, 30.842317360],
        "qingdao": [119.974220271, 120.481110190, 36.044704313, 36.403399024],
    }

    def func1():

        def dirname_imdc_fn(_dirname):
            for fn in os.listdir(_dirname):
                if fn.endswith("_imdc.tif"):
                    return os.path.join(_dirname, fn)

        raster_fns = {
            "QingDao_ML_VHL3": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240805H105502"),
            "BeiJing_ML_VHL3": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H094617"),
            "ChengDu_ML_VHL3": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H101049"),
            "QingDao_ML_Category4": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H101844"),
            "BeiJing_ML_Category4": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H102411"),
            "ChengDu_ML_Category4": dirname_imdc_fn(r"F:\ProjectSet\Shadow\Hierarchical\GDMLMods\20240806H102804"),
        }

        def select_samples(df, n, cname, *filters):
            spls = DFF_AND(df, *filters).to_dict("records")
            if n != -1:
                if len(spls) > n:
                    spls = random.sample(spls, n)
                else:
                    print("- {}<{} {} {}".format(len(spls), n, cname, filters))
            for i in range(len(spls)):
                spls[i]["CNAME"] = cname
                spls[i]["TEST"] = int(random.random() < 0.9)
            return spls

        def qd():

            df = pd.DataFrame(randomSamples(100000, *region_dict["qingdao"]))
            df = GDALSamplingFast(
                raster_fn=raster_fns["QingDao_ML_VHL3"]
            ).samplingDF(df).rename(columns={"FEATURE_1": "VHL3"})
            df = GDALSamplingFast(
                raster_fn=raster_fns["QingDao_ML_Category4"]
            ).samplingDF(df).rename(columns={"FEATURE_1": "Category4"})

            print(df)
            dfnumber(df, "VHL3", "Category4")

            to_list = [
                *select_samples(df, 2000, "IS", ("VHL3", "==", 1), ("Category4", "==", 1)),
                *select_samples(df, 300, "VEG", ("VHL3", "==", 1), ("Category4", "==", 2)),
                *select_samples(df, 1500, "SOIL", ("VHL3", "==", 1), ("Category4", "==", 3)),
                *select_samples(df, -1, "WAT", ("VHL3", "==", 1), ("Category4", "==", 4)),

                *select_samples(df, -1, "IS", ("VHL3", "==", 2), ("Category4", "==", 1)),
                *select_samples(df, 2000, "VEG", ("VHL3", "==", 2), ("Category4", "==", 2)),
                *select_samples(df, -1, "SOIL", ("VHL3", "==", 2), ("Category4", "==", 3)),
                *select_samples(df, -1, "WAT", ("VHL3", "==", 2), ("Category4", "==", 4)),

                *select_samples(df, 1000, "IS_SH", ("VHL3", "==", 3), ("Category4", "==", 1)),
                *select_samples(df, 1000, "VEG_SH", ("VHL3", "==", 3), ("Category4", "==", 2)),
                *select_samples(df, -1, "SOIL_SH", ("VHL3", "==", 3), ("Category4", "==", 3)),
                *select_samples(df, 1000, "WAT", ("VHL3", "==", 3), ("Category4", "==", 4)),

            ]

            to_df = pd.DataFrame(to_list)
            to_df["CATEGORY"] = to_df["VHL3"] * 10 + to_df["Category4"]
            print(to_df)
            to_df.to_csv(W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd4.csv"), index=False)

        def bj():
            df = pd.DataFrame(randomSamples(100000, *region_dict["beijing"]))
            df = GDALSamplingFast(
                raster_fn=raster_fns["BeiJing_ML_VHL3"]
            ).samplingDF(df).rename(columns={"FEATURE_1": "VHL3"})
            df = GDALSamplingFast(
                raster_fn=raster_fns["BeiJing_ML_Category4"]
            ).samplingDF(df).rename(columns={"FEATURE_1": "Category4"})

            print(df)
            dfnumber(df, "VHL3", "Category4")

            to_list = [
                *select_samples(df, 2000, "IS", ("VHL3", "==", 1), ("Category4", "==", 1)),
                *select_samples(df, 300, "VEG", ("VHL3", "==", 1), ("Category4", "==", 2)),
                *select_samples(df, 1500, "SOIL", ("VHL3", "==", 1), ("Category4", "==", 3)),
                *select_samples(df, -1, "WAT", ("VHL3", "==", 1), ("Category4", "==", 4)),

                *select_samples(df, -1, "IS", ("VHL3", "==", 2), ("Category4", "==", 1)),
                *select_samples(df, 2000, "VEG", ("VHL3", "==", 2), ("Category4", "==", 2)),
                *select_samples(df, -1, "SOIL", ("VHL3", "==", 2), ("Category4", "==", 3)),
                *select_samples(df, -1, "WAT", ("VHL3", "==", 2), ("Category4", "==", 4)),

                *select_samples(df, 1000, "IS_SH", ("VHL3", "==", 3), ("Category4", "==", 1)),
                *select_samples(df, 1000, "VEG_SH", ("VHL3", "==", 3), ("Category4", "==", 2)),
                *select_samples(df, -1, "SOIL_SH", ("VHL3", "==", 3), ("Category4", "==", 3)),
                *select_samples(df, 1000, "WAT", ("VHL3", "==", 3), ("Category4", "==", 4)),

            ]

            to_df = pd.DataFrame(to_list)
            to_df["CATEGORY"] = to_df["VHL3"] * 10 + to_df["Category4"]
            print(to_df)
            to_df.to_csv(W2LF(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj3.csv"), index=False)

        return bj()

    def func2():
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']

        CNAME_COLORS = {
            "IS": (255, 0, 0), "VEG": (0, 255, 0), "SOIL": (255, 255, 0), "WAT": (0, 0, 255),
            "IS_SH": (128, 0, 0), "VEG_SH": (0, 128, 0), "SOIL_SH": (128, 128, 0), "WAT_SH": (0, 0, 128)
        }
        CNAME_COLORS = {name: to_hex(
            (CNAME_COLORS[name][0] / 255, CNAME_COLORS[name][1] / 255, CNAME_COLORS[name][2] / 255)
        ) for name in CNAME_COLORS}

        fig = plt.figure(figsize=(10, 8), )
        fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.80, hspace=0.03, wspace=0.03)

        # export to no look file
        def nolook(to_df, look_csv_fn):
            look_df = to_df[to_df["LOOK"] == 0]
            look_df.to_csv(look_csv_fn, index=False)
            SAMPLING_CITY_NAME("bj", look_csv_fn)
            look_df = pd.read_csv(look_csv_fn)
            look_df = look_df.sort_values(by=["CNAME", "NDVI"], ascending=[True, False])
            look_df["CATEGORY"] = mapDict(look_df["CNAME"].to_list(), SHHConfig.CNAME_MAP_SH882)
            look_df.to_csv(look_csv_fn, index=False)

        def qd():
            shh2_sn = SHH2SamplesNumbers(
                end_cnames=["IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH", ])
            shh2_sn.add("ISO", "IS", IS=1022)
            shh2_sn.add("ISO", "SOIL", SOIL=1012)
            shh2_sn.add("WS", "IS_SH", IS_SH=586)
            shh2_sn.add("WS", "NOIS_SH", VEG_SH=516, SOIL_SH=53)
            shh2_sn.add("WS", "WAT", WAT=665, WAT_SH=56)
            shh2_sn.add("VHL", "HIGH", IS=813, SOIL=324)
            shh2_sn.add("VHL", "VEG", VEG=1039)
            shh2_sn.add("VHL", "LOW", WAT=313, IS_SH=203, VEG_SH=188, SOIL_SH=23, WAT_SH=25)

            # add to csv files
            csv_fns = []
            # original samples csv files
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd1.csv"
            df = pd.read_csv(csv_fn)
            shh2_sn.addSamples(df[df["TEST"] == 1].to_dict("records"), {"LOOK": 1, "SOURCE": csv_fn})
            csv_fns.append(csv_fn)
            # 10w random from func1() samples
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd5_nolook22.csv"
            df_nolook = pd.read_csv(csv_fn).to_dict("records")
            shh2_sn.addSamples(df_nolook, {"LOOK": 1, "SOURCE": csv_fn, "TEST": 1})
            csv_fns.append(csv_fn)
            # select
            txt_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd7.txt"
            shh2_sn.addSamplesQJY(txt_fn, {"LOOK": 1, "TEST": 1})
            # shadow 1 samples
            csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd8.csv"
            df_nolook = pd.read_csv(csv_fn).to_dict("records")
            shh2_sn.addSamples(df_nolook, {"LOOK": 1, "SOURCE": csv_fn, "TEST": 1})
            csv_fns.append(csv_fn)

            shh2_sn.show("current")
            shh2_sn.scatter(color_dict=CNAME_COLORS)

            # save samples to csv file
            to_csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd6.csv"
            to_df = shh2_sn.toCSV(
                to_csv_fn,
                select_names=["X", "Y", "CNAME", "FCNAME", "SUBNAME", "TEST", "OSRT", "LOOK", "SOURCE"],
                samples=df[df["TEST"] == 0].to_dict("records")
            )
            to_df = pd.read_csv(to_csv_fn)

            print("# LOOK TEST ------")
            dfnumber(to_df, "CNAME", "LOOK")
            print("# FCNAME ------")
            dfnumber(to_df, "CNAME", "FCNAME")

            # nolook(to_df, r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd5_nolook.csv")

        def bj():
            dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj")
            shh2_sn = SHH2SamplesNumbers(
                end_cnames=["IS", "IS_SH", "VEG", "VEG_SH", "SOIL", "SOIL_SH", "WAT", "WAT_SH", ])
            shh2_sn.add("ISO", "IS", IS=1022)
            shh2_sn.add("ISO", "SOIL", SOIL=1012)
            shh2_sn.add("WS", "IS_SH", IS_SH=586)
            shh2_sn.add("WS", "NOIS_SH", VEG_SH=516, SOIL_SH=53)
            shh2_sn.add("WS", "WAT", WAT=665, WAT_SH=56)
            shh2_sn.add("VHL", "HIGH", IS=813, SOIL=324)
            shh2_sn.add("VHL", "VEG", VEG=1039)
            shh2_sn.add("VHL", "LOW", WAT=313, IS_SH=203, VEG_SH=188, SOIL_SH=23, WAT_SH=25)

            def tiaoshi():
                # original samples csv files
                csv_fn = dfn.fn("sh2_spl30_bj1.csv")
                df = pd.read_csv(csv_fn)
                shh2_sn.addSamples(df[df["TEST"] == 1].to_dict("records"), {"LOOK": 1, "SOURCE": csv_fn})
                # shadow 1 samples
                csv_fn = dfn.fn("sh2_spl30_bj43.csv")
                df_sh1 = pd.read_csv(csv_fn)
                df_sh1_list = samplesFilterOR(
                    df_sh1.to_dict("records"),
                    # ("CNAME", "==", "IS_SH"),
                    # ("CNAME", "==", "VEG_SH"),
                    # ("CNAME", "==", "SOIL_SH"),
                    # ("CNAME", "==", "WAT_SH"),
                    # ("CNAME", "==", "WAT"),
                    # ("CNAME", "==", "SOIL"),
                )
                shh2_sn.addSamples(df_sh1_list, {"LOOK": 1, "SOURCE": csv_fn, "TEST": 1})
                # 10w random from func1() samples
                csv_fn = dfn.fn("sh2_spl30_bj3.csv")
                df_nolook = pd.read_csv(csv_fn).to_dict("records")
                shh2_sn.addSamples(df_nolook, {"LOOK": 0, "SOURCE": csv_fn, "TEST": 1})
                # # select
                # txt_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd7.txt"
                # shh2_sn.addSamplesQJY(txt_fn, {"LOOK": 1, "TEST": 1})
                _to_df = shh2_sn.toCSV(
                    to_csv_fn,
                    select_names=["X", "Y", "CNAME", "FCNAME", "SUBNAME", "TEST", "OSRT", "LOOK", "SOURCE"],
                    samples=df[df["TEST"] == 0].to_dict("records")
                )
                return _to_df

            def fubu():
                csv_fn = dfn.fn("sh2_spl30_bj6.csv")
                df = pd.read_csv(csv_fn)
                shh2_sn.addSamples(df[df["TEST"] == 1].to_dict("records"), {"SOURCE": csv_fn})
                _to_df = shh2_sn.toCSV(
                    to_csv_fn,
                    select_names=["X", "Y", "CNAME", "FCNAME", "SUBNAME", "TEST", "OSRT", "LOOK", "SOURCE"],
                    samples=df[df["TEST"] == 0].to_dict("records")
                )
                return _to_df

            # save samples to csv file
            to_csv_fn = dfn.fn("sh2_spl30_bj.csv")
            fubu()
            shh2_sn.show("current")
            shh2_sn.scatter(color_dict=CNAME_COLORS)

            to_df = pd.read_csv(to_csv_fn)

            print("# LOOK ------")
            dfnumber(to_df, "CNAME", "LOOK")
            print("# FCNAME ------")
            dfnumber(to_df, "CNAME", "FCNAME")

            nolook(to_df, dfn.fn("sh2_spl30_bj_look.csv"))

        bj()

        plt.show()

        return

    def func3():
        shh2ShowCSVNumbers(
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd1.csv",
            r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\qd\sh2_spl30_qd5_nolook22.csv",
            name1="CNAME", name2="TEST"
        )

    def func4():
        def func41():
            df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj4.csv")
            samples = uniqueSamples(SHH2Config.BJ_ENVI_FN, df[df["TEST"] == 1].to_dict("records"))
            samples.extend(df[df["TEST"] == 0].to_dict("records"))
            df = pd.DataFrame(samples)
            df.to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj41.csv", index=False)

        def func42():
            x1, x2, y1, y2 = region_dict["beijing"]
            df_list = pd.read_csv(
                r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj42.csv").to_dict(
                "records")
            to_list = []
            for spl in df_list:
                if (x1 < spl["X"] < x2) and (y1 < spl["Y"] < y2):
                    to_list.append(spl)
            pd.DataFrame(to_list).to_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj43.csv",
                                         index=False)

        def func43():
            df = pd.read_csv(r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\bj\sh2_spl30_bj43.csv")
            dfnumber(df, "CNAME", "TEST")

        return func43()

    return func2()


if __name__ == "__main__":
    samplesFind()


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
