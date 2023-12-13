# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowBack.py.py
@Time    : 2023/7/1 14:35
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ShadowBack.py
-----------------------------------------------------------------------------"""
import json
import os
import struct
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from SRTCodes.NumpyUtils import saveCM

pd.set_option('display.width', 500)  # 数据显示总宽度
pd.set_option('display.max_columns', 10)  # 显示最多列数，超出该数以省略号表示
pd.set_option('display.max_colwidth', 16)  # 设置单列的宽度，用字符个数表示，单个数据长度超出该数时以省略号表示
np.set_printoptions(linewidth=500, precision=8, suppress=True)

with open(r"ASDESH_Spl1001_Feat9_Algo2.json", "r", encoding="utf-8") as f:
    save_d = json.load(f)
# 数据中所有特征的名字
feat_names_all = np.array(save_d["feat_names_all"])
# 数据伸缩的参数
re_hist = np.array(save_d["re_hist"])
# 特征类型的名称
feat_type_names = np.array(save_d["feat_type_names"])
# ENVI Class 文件信息
envi_class_file_info = save_d["envi_class_file_info"]

TEMP_DIR = time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
MODEL_DIR = r"F:\ProjectSet\Shadow\QingDao\Mods"


def plotImageHist(imd: np.array, bands=None, labels=None):
    """ 绘制图像的波段数据分布图

    :param imd: 图像数据
    :param bands: 绘制的波段
    :param labels: 显示的标签
    :return:
    """
    if bands is None:
        bands = [i for i in range(imd.shape[0])]
    if labels is None:
        labels = ["Band " + str(i + 1) for i in range(imd.shape[0])]

    for i, i_band in enumerate(bands):
        h, bin_edges = np.histogram(imd[i_band], bins=256)
        plt.plot(h, label=labels[i])

    plt.legend()
    plt.show()


def calCM(in_cm: np.array):
    """ 混淆矩阵

    :param in_cm: 输入混淆矩阵
    :return: 混淆矩阵
    """
    n_class = in_cm.shape[0]
    out_cm = np.zeros([n_class + 2, n_class + 2])
    out_cm[:n_class, :n_class] = in_cm
    out_cm[n_class, :] = np.sum(out_cm, axis=0)
    out_cm[:, n_class] = np.sum(out_cm, axis=1)
    out_cm[n_class + 1, :] = np.diag(out_cm) * 1.0 / (out_cm[n_class, :] + 0.0000001) * 100
    out_cm[:, n_class + 1] = np.diag(out_cm) * 1.0 / (out_cm[:, n_class] + 0.0000001) * 100
    out_cm[n_class + 1, n_class + 1] = (np.sum(np.diag(in_cm))) / out_cm[n_class, n_class] * 100
    return out_cm


class EnviFileIO:
    """

    """

    def __init__(self, im_file=None, im_info_json_file=""):
        self.im_file = im_file
        self.im_infos = None
        if im_info_json_file == "" and im_file is not None:
            self.setInfoFromHDRFile(os.path.splitext(im_file)[0] + ".hdr")
        else:
            if im_info_json_file != "":
                self.setInfoFromJsonFile(im_info_json_file)
        self.d = None
        pass

    def setInfoFromJsonFile(self, json_file):
        """ 使用json文件获得保存影像的信息

        :param json_file: json文件
        :return: 信息
        """
        self.im_infos = None
        with open(json_file, "r", encoding="utf-8") as fr:
            self.im_infos = json.load(fr)
        return self.im_infos

    def setInfoFromHDRFile(self, hdr_file):
        """ 使用hdr文件获得保存影像的信息

        :param hdr_file: hdr文件
        :return: 信息
        """
        lines = []
        with open(hdr_file, "r", encoding="utf-8") as fr:
            for line in fr:
                if line.strip() == "ENVI":
                    continue
                if line.find("=") != -1:
                    lines.append(line)
                else:
                    lines[-1] += line
        self.im_infos = {}
        for line in lines:
            line1 = line.split("=", 2)
            self.im_infos[line1[0].strip()] = line1[1].strip()
        return self.im_infos

    def readToArray(self, interleave="r,c,b"):
        """
        :return:
        """
        n_rows = int(self.im_infos["lines"])
        n_columns = int(self.im_infos["samples"])
        n_bands = int(self.im_infos["bands"])
        n_data = n_rows * n_columns * n_bands
        with open(self.im_file, "rb") as frb:
            if self.im_infos["data type"] == "1":
                self.d = struct.unpack(str(n_data) + "B", frb.read())
            elif self.im_infos["data type"] == "2":
                self.d = struct.unpack(str(n_data) + "h", frb.read())
            elif self.im_infos["data type"] == "3":
                self.d = struct.unpack(str(n_data) + "i", frb.read())
            elif self.im_infos["data type"] == "4":
                self.d = struct.unpack(str(n_data) + "f", frb.read())
            elif self.im_infos["data type"] == "5":
                self.d = struct.unpack(str(n_data) + "d", frb.read())
            elif self.im_infos["data type"] == "6":
                self.d = struct.unpack(str(n_data * 2) + "f", frb.read())
            elif self.im_infos["data type"] == "9":
                self.d = struct.unpack(str(n_data * 2) + "d", frb.read())
            elif self.im_infos["data type"] == "12":
                self.d = struct.unpack(str(n_data) + "H", frb.read())
            elif self.im_infos["data type"] == "13":
                self.d = struct.unpack(str(n_data * 2) + "I", frb.read())
            elif self.im_infos["data type"] == "14":
                self.d = struct.unpack(str(n_data * 2) + "q", frb.read())
            elif self.im_infos["data type"] == "15":
                self.d = struct.unpack(str(n_data * 2) + "Q", frb.read())
            else:
                raise Exception("Can not find \"data type\"=" + self.im_infos["data type"])
        self.d = np.array(self.d)
        if self.im_infos["interleave"].lower() == "bsq":
            self.d = self.d.reshape([n_bands, n_rows, n_columns])
            if interleave == "r,c,b":
                self.d = np.transpose(self.d, axes=(1, 2, 0))
        elif self.im_infos["interleave"].lower() == "bip":
            self.d = self.d.reshape([n_rows, n_columns, n_bands])
            if interleave == "b,r,c":
                self.d = np.transpose(self.d, axes=(2, 0, 1))
        elif self.im_infos["interleave"].lower() == "bil":
            self.d = self.d.reshape([n_rows, n_bands, n_columns])
            if interleave == "b,r,c":
                self.d = np.transpose(self.d, axes=(1, 0, 2))
            if interleave == "r,c,b":
                self.d = np.transpose(self.d, axes=(0, 2, 1))
        return self.d

    def infoToJson(self, json_file):
        with open(json_file, "w", encoding="utf-8") as fw:
            json.dump(self.im_infos, fw)
        return self.im_infos

    def saveToFile(self, imd, im_infos=None, out_file=None):
        if out_file is None:
            out_file = self.im_file
        if im_infos is None:
            im_infos = self.im_infos
        to_f = os.path.splitext(out_file)[0]
        with open(to_f + ".hdr", "w", encoding="utf-8") as f:
            f.write("ENVI\n")
            for k in im_infos:
                f.write(k + " = " + im_infos[k] + "\n")
        with open(out_file, "wb") as f:
            imd.tofile(f)
        return im_infos

    @classmethod
    def hdr2ImInfo(cls, hdr_file, im_info_json_file):
        """ ENVI头文件转信息文件

        :param hdr_file: ENVI头文件
        :param im_info_json_file:信息文件
        :return:
        """
        lines = []
        with open(hdr_file, "r", encoding="utf-8") as fr:
            for line in fr:
                if line.strip() == "ENVI":
                    continue
                if line.find("=") != -1:
                    lines.append(line)
                else:
                    lines[-1] += line
        infos = {}
        for line in lines:
            line1 = line.split("=", 2)
            infos[line1[0].strip()] = line1[1].strip()
        with open(im_info_json_file, "w", encoding="utf-8") as fw:
            json.dump(infos, fw)
        return infos


class Jdt:
    """
    进度条
    """

    def __init__(self, total=100, desc=None, iterable=None, n_cols=20):
        """ 初始化一个进度条对象

        :param iterable: 可迭代的对象, 在手动更新时不需要进行设置
        :param desc: 字符串, 左边进度条描述文字
        :param total: 总的项目数
        :param n_cols: 调整进度条宽度, 默认是根据环境自动调节长度, 如果设置为0, 就没有进度条, 只有输出的信息
        """
        self.total = total
        self.iterable = iterable
        self.n_cols = n_cols
        self.desc = desc if desc is not None else ""

        self.n_split = float(total) / float(n_cols)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False

    def start(self):
        """ 开始进度条 """
        self.is_run = True
        print()

    def add(self, n=1):
        """ 添加n个进度

        :param n: 进度的个数
        :return:
        """
        if self.is_run:
            self.n_current += n
            if self.n_current > self.n_print * self.n_split:
                self.n_print += 1
                if self.n_print > self.n_cols:
                    self.n_print = self.n_cols
            self._print()

    def setDesc(self, desc):
        """ 添加打印信息 """
        self.desc = desc

    def _print(self):
        des_info = "\r{0}: {1:>3d}% |".format(self.desc, int(self.n_current / self.total * 100))
        des_info += "*" * self.n_print + "-" * (self.n_cols - self.n_print)
        des_info += "| {0}/{1}".format(self.n_current, self.total)
        print(des_info, end="")

    def end(self):
        """ 结束进度条 """
        self.n_split = float(self.total) / float(self.n_split)
        self.n_current = 0
        self.n_print = 0
        self.is_run = False
        print()


class RumTime:

    def __init__(self, n_all=0):
        self.n_all = n_all
        self.n_current = 0
        self.strat_time = time.time()
        self.current_time = time.time()

    def strat(self):
        self.n_current = 0
        self.strat_time = time.time()
        self.current_time = time.time()

    def add(self, n=1):
        self.n_current += 1
        self.current_time = time.time()

    def printInfo(self):
        out_s = f"+ {self.n_current}"
        # time.strftime('%Y-%m-%d %H-%M-%S', time.localtime())
        out_s += " RUN:"
        out_s += RumTime.fmtTime(self.current_time - self.strat_time)
        if self.n_all != 0:
            out_s += " ALL:"
            t1 = (self.current_time - self.strat_time) / (self.n_current + 0.0000001) * self.n_all
            out_s += RumTime.fmtTime(t1)
        print(out_s)

    def end(self):
        print("end")

    @classmethod
    def fmtTime(cls, t):
        hours = t // 3600
        minutes = (t - 3600 * hours) // 60
        seconds = t - 3600 * hours - minutes * 60
        return f"({int(hours)}:{int(minutes)}:{seconds:.2f})"


class SRTFeature:
    """
    获得不同特征的名字
    """

    def __init__(self):
        self.is_end = False
        self.i_feat_types = []  # 特征类型组合的索引
        with open(r"full_extraction.txt", "r", encoding="utf-8") as f:
            for line in f:
                lines = line.split(" ")
                self.i_feat_types.append(list(map(eval, lines)))
                self.i_feat_types[-1] = list(map(lambda x: x - 1, self.i_feat_types[-1]))
        self.i_current = 0
        self.feat_types = []

    def get(self):
        if self.i_current < len(self.i_feat_types):
            self.feat_types = list(feat_type_names[self.i_feat_types[self.i_current]])
            self.i_current += 1
            return self.get_feat_names()
        else:
            self.feat_types = []
            return None

    def clear(self):
        self.i_current = 0

    def get_feat_names(self, return_names=False):
        feat_index = []
        for feat_type in self.feat_types:
            if feat_type == "RGBN":  # 1 可见光和近红外
                feat_index.extend(['B', 'G', 'R', 'N'])
            elif feat_type == "SI":  # 2 光谱指数
                feat_index.extend(['NDVI', 'NDWI'])
            elif feat_type == "ASVVVH":  # 3 升轨VV和VH
                feat_index.extend(['VV_AS', 'VH_AS'])
            elif feat_type == "DEVVVH":  # 4 降轨VV和VH
                feat_index.extend(['VV_DE', 'VH_DE'])
            elif feat_type == "ASVVVHGLCM":  # 5 升轨VV和VH纹理
                feat_index.extend(['VH_AS_Mean', 'VH_AS_Variance', 'VH_AS_Homogeneity', 'VV_AS_Mean', 'VV_AS_Variance',
                                   'VV_AS_Homogeneity'])
            elif feat_type == "DEVVVHGLCM":  # 6 降轨VV和VH纹理
                feat_index.extend(['VH_DE_Mean', 'VH_DE_Variance', 'VH_DE_Homogeneity', 'VV_DE_Mean', 'VV_DE_Variance',
                                   'VV_DE_Homogeneity'])
            elif feat_type == "PCGLCM":  # 7 降轨VV和VH纹理
                feat_index.extend(['VH_DE_Mean', 'VH_DE_Variance', 'VH_DE_Homogeneity', 'VV_DE_Mean', 'VV_DE_Variance',
                                   'VV_DE_Homogeneity'])
            elif feat_type == "CAS":  # 8 升轨协方差矩阵
                feat_index.extend(['AS_20210507_C22', 'AS_20210507_C12real', 'AS_20210507_C12imag', 'AS_20210507_C11'])
            elif feat_type == "CDE":  # 9 升轨协方差矩阵
                feat_index.extend(['DE_20210430_C22', 'DE_20210430_C12real', 'DE_20210430_C12imag', 'DE_20210430_C11'])
            else:
                raise Exception("Not find feature type as " + feat_type)
        if return_names:
            return "_".join(feat_index), feat_index
        else:
            return feat_index


class SRTSample:

    def __init__(self, train_csv_fn, test_csv_fn=None):
        """ 样本管理工具

        :param train_csv_fn: 训练数据csv问价
        :param test_csv_fn: 测试数据csv文件
        """
        self.test_feat_names = None
        self.train_feat_names = None
        self.d_o_test = None
        self.d_o_train = None
        self.train_csv_fn = train_csv_fn
        self.test_csv_fn = test_csv_fn
        self.feat_names = []
        self.train_cate_codes = None
        self.test_cate_codes = None
        self.d_o = None
        self.d_train = None
        self.d_test = None
        self.train_test = None
        if self.test_csv_fn is None:
            self.initFromCsv_train(self.train_csv_fn)
        else:
            self.initFromCsv_traintest(self.train_csv_fn, self.test_csv_fn)

    def initFromCsv_train(self, train_csv_file):
        self.d_o = pd.read_csv(train_csv_file)
        self.train_test = self.d_o["TrainOrTest"].values
        # self.train_test = (np.random.random(len(self.d_o)) < 0.7).astype("int")
        self.feat_names = list(self.d_o.keys())
        self.train_cate_codes = self.d_o["CATEGORY"].values[self.train_test == 1]
        self.test_cate_codes = self.d_o["CATEGORY"].values[self.train_test == 0]
        self.feat_names = self.feat_names[5:]
        self.d_train = np.zeros([len(self.d_o), len(self.feat_names)])
        for i, k in enumerate(self.feat_names):
            self.d_train[:, i] = np.clip(self.d_o[k].values, re_hist[i, 0], re_hist[i, 1])
            self.d_train[:, i] = (self.d_train[:, i] - re_hist[i, 0]) / (re_hist[i, 1] - re_hist[i, 0])
        self.d_test = pd.DataFrame(data=self.d_train[self.train_test == 0], columns=self.feat_names)
        self.d_train = pd.DataFrame(data=self.d_train[self.train_test == 1], columns=self.feat_names)

    def initFromCsv_traintest(self, train_csv_fn, test_csv_fn):
        self.d_o_train = pd.read_csv(train_csv_fn)
        self.train_cate_codes = self.d_o_train["CATEGORY"].values
        self.train_feat_names = list(self.d_o_train.keys())[5:]
        self.d_train = np.zeros([len(self.d_o_train), len(self.train_feat_names)])
        for i, k in enumerate(self.train_feat_names):
            self.d_train[:, i] = np.clip(self.d_o_train[k].values, re_hist[i, 0], re_hist[i, 1])
            self.d_train[:, i] = (self.d_train[:, i] - re_hist[i, 0]) / (re_hist[i, 1] - re_hist[i, 0])
        self.d_train = pd.DataFrame(data=self.d_train, columns=self.train_feat_names)
        self.d_o_test = pd.read_csv(test_csv_fn)
        self.test_cate_codes = self.d_o_test["CATEGORY"].values
        self.test_feat_names = list(self.d_o_test.keys())[5:]
        self.d_test = np.zeros([len(self.d_o_test), len(self.test_feat_names)])
        for i, k in enumerate(self.test_feat_names):
            self.d_test[:, i] = np.clip(self.d_o_test[k].values, re_hist[i, 0], re_hist[i, 1])
            self.d_test[:, i] = (self.d_test[:, i] - re_hist[i, 0]) / (re_hist[i, 1] - re_hist[i, 0])
        self.d_test = pd.DataFrame(data=self.d_test, columns=self.test_feat_names)
        pass

    def get(self, spl_type, feat_index):
        """ 使用样本类型和特征类型作为样本获取标识

        :param spl_type: 样本类型 SPLS, NOSPLS
        :param feat_index: 特征类型
        :return: 样本和标签
        """
        d_index = np.array([True for i in range(len(self.d_train))])
        if spl_type == "NOSPLS":
            d_index = (self.train_cate_codes == 11) + (self.train_cate_codes == 21) \
                      + (self.train_cate_codes == 31) + (self.train_cate_codes == 41)
        d_train: pd.DataFrame = self.d_train[feat_index][d_index]
        labels_train: np.ndarray = self.train_cate_codes[d_index]
        labels_train = np.floor(labels_train / 10)
        d_test: pd.DataFrame = self.d_test[feat_index]
        labels_test: np.ndarray = self.test_cate_codes
        return d_train, labels_train, d_test, labels_test


class SRTClassAlgos:

    def __init__(self):
        self.cm = None
        self.test_acc = None
        self.train_acc = None
        self.model = None

    def SVM(self, x, y, x_test, y_test, save_file=None):
        """ SVM 分类器

        :param x: 训练数据
        :param y: 训练数据标签
        :param x_test: 测试数据
        :param y_test: 测试数据标签
        :param save_file: 保存的文件
        :return:
        """
        svm_args = {"kernel": "rbf", "gamma": "auto", "C": 1}
        refer_args = {}
        svm_args, refer_args = SRTClassAlgos.trainSvm(y, x)
        algo_infos = {
            "algo_type": "RF",
            "algo_args": svm_args,
            "train_accuracy": 0,
            "test_accuracy": 0,
            "train_cm": None,
            "test_cm": None,
            "refer_args": refer_args
        }
        self.model = SVC(
            kernel=svm_args["kernel"],
            C=svm_args["C"],
            gamma=svm_args["gamma"],
            random_state=90)

        self.model.fit(x, y)
        self.train_acc = self.model.score(x, y)
        self.test_acc = self.model.score(x_test, y_test)
        algo_infos["train_accuracy"] = self.train_acc
        algo_infos["test_accuracy"] = self.test_acc
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.model.predict(x_test))
        self.cm = calCM(self.cm)
        train_cm = confusion_matrix(y_true=y, y_pred=self.model.predict(x))
        algo_infos["train_cm"] = calCM(train_cm).tolist()
        algo_infos["test_cm"] = self.cm.tolist()
        if save_file is not None:
            joblib.dump(self.model, save_file)
        return algo_infos

    def RF(self, x, y, x_test, y_test, save_file=None):
        """ Random Forest 分类器

        :param x: 训练数据
        :param y: 训练数据标签
        :param x_test: 测试数据
        :param y_test: 测试数据标签
        :param save_file: 保存的文件
        :return:
        """
        rf_args = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 1, "min_samples_split": 18}
        refer_args = {}
        rf_args, refer_args = SRTClassAlgos.trainRF(y, x)
        algo_infos = {
            "algo_type": "RF",
            "algo_args": rf_args,
            "train_accuracy": 0,
            "test_accuracy": 0,
            "train_cm": None,
            "test_cm": None,
            "refer_args": refer_args
        }
        self.model = RandomForestClassifier(
            n_estimators=rf_args["n_estimators"],
            max_depth=rf_args["max_depth"],
            min_samples_leaf=rf_args["min_samples_leaf"],
            min_samples_split=rf_args["min_samples_split"],
            random_state=90)
        self.model.fit(x, y)
        self.train_acc = self.model.score(x, y)
        self.test_acc = self.model.score(x_test, y_test)
        algo_infos["train_accuracy"] = self.train_acc
        algo_infos["test_accuracy"] = self.test_acc
        self.cm = confusion_matrix(y_true=y_test, y_pred=self.model.predict(x_test))
        self.cm = calCM(self.cm)
        train_cm = confusion_matrix(y_true=y, y_pred=self.model.predict(x))
        algo_infos["train_cm"] = calCM(train_cm).tolist()
        algo_infos["test_cm"] = self.cm.tolist()
        if save_file is not None:
            joblib.dump(self.model, save_file)
        return algo_infos

    @classmethod
    def trainRF(cls, labels, d_train):
        rf_args = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 1, "min_samples_split": 18}
        refer_args_infos = {}
        # Tuning parameters: n_estimators -----------------------------------------
        print("n_estimators: ", end="")
        scorel, s_max, canshu = [], 0, list(range(1, 150, 10))
        for i in canshu:
            rfc = RandomForestClassifier(n_estimators=i, n_jobs=-1, random_state=90)
            score = cross_val_score(rfc, d_train, labels, cv=10).mean()
            scorel.append(score)
            print(f"{i}:{score * 100:.2f}", end=" ")
            if score > s_max:
                s_max = score
                rf_args["n_estimators"] = i
        refer_args_infos["n_estimators"] = {"accuracy": scorel, "args": canshu}
        print("\n  -> ", max(scorel) * 100, rf_args["n_estimators"])
        # Tuning parameters: max_depth --------------------------------------------
        print("max_depth: ", end="")
        scorel, s_max, canshu = [], 0, list(range(1, 20))
        for i in canshu:
            rfc = RandomForestClassifier(
                n_estimators=rf_args["n_estimators"]
                , max_depth=i
                , n_jobs=-1, random_state=90)
            score = cross_val_score(rfc, d_train, labels, cv=10).mean()
            print(f"{i}:{score * 100:.2f}", end=" ")
            scorel.append(score)
            if score > s_max:
                s_max = score
                rf_args["max_depth"] = i
        print("\n  -> ", max(scorel), rf_args["max_depth"])
        refer_args_infos["max_depth"] = {"accuracy": scorel, "args": canshu}
        # Tuning parameters: min_samples_leaf -------------------------------------
        print("min_samples_leaf: ", end="")
        scorel, s_max, canshu = [], 0, list(range(1, 5))
        for i in canshu:
            rfc = RandomForestClassifier(
                n_estimators=rf_args["n_estimators"]
                , max_depth=rf_args["max_depth"]
                , min_samples_leaf=i
                , n_jobs=-1, random_state=90)
            score = cross_val_score(rfc, d_train, labels, cv=10).mean()
            print(f"{i}:{score * 100:.2f}", end=" ")
            scorel.append(score)
            if score > s_max:
                s_max = score
                rf_args["min_samples_leaf"] = i
        print("\n  -> ", max(scorel), rf_args["min_samples_leaf"])
        refer_args_infos["min_samples_leaf"] = {"accuracy": scorel, "args": canshu}
        # Tuning parameters: min_samples_split ------------------------------------
        print("min_samples_split: ", end="")
        scorel, s_max, canshu = [], 0, list(range(2, 10))
        for i in canshu:
            rfc = RandomForestClassifier(
                n_estimators=rf_args["n_estimators"]
                , max_depth=rf_args["max_depth"]
                , min_samples_leaf=rf_args["min_samples_leaf"]
                , min_samples_split=i
                , n_jobs=-1, random_state=90)
            score = cross_val_score(rfc, d_train, labels, cv=10).mean()
            print(f"{i}:{score * 100:.2f}", end=" ")
            scorel.append(score)
            if score > s_max:
                s_max = score
                rf_args["min_samples_split"] = i
        print("\n  -> ", max(scorel), rf_args["min_samples_split"])
        refer_args_infos["min_samples_split"] = {"accuracy": scorel, "args": canshu}
        return rf_args, refer_args_infos

    @classmethod
    def trainSvm(cls, labels, d_train):
        svm_args = {"kernel": "rbf", "gamma": "auto", "C": 1}
        refer_args_infos = {}
        # 调线软间隔C
        print("C: ", end="")
        s_max, scores = 0, []
        C_range = np.linspace(0.01, 10, 20)
        for i in C_range:
            clf = SVC(
                kernel=svm_args["kernel"],
                C=i,
                cache_size=5000)
            score = cross_val_score(clf, d_train, labels, cv=10).mean()
            scores.append(score)
            print(f"{i:.3f}:{scores[-1] * 100:.2f}", end=" ")
            if scores[-1] > s_max:
                s_max = scores[-1]
                svm_args["C"] = i
        refer_args_infos["C"] = {"accuracy": scores, "args": C_range.tolist()}
        print("\n  -> ", s_max, svm_args["C"])
        # plt.close()
        # plt.plot(C_range, scores)
        # plt.savefig("../Data/C.png")
        # 调 gamma
        print("gamma: ", end="")
        s_max, scores = 0, []
        gamma_range = np.logspace(-1, 1, 20)
        for i in gamma_range:
            clf = SVC(
                kernel=svm_args["kernel"],
                C=svm_args["C"],
                gamma=i,
                cache_size=5000)
            score = cross_val_score(clf, d_train, labels, cv=10).mean()
            scores.append(score)
            print(f"{i:.3f}:{scores[-1] * 100:.2f}", end=" ")
            if scores[-1] > s_max:
                s_max = scores[-1]
                svm_args["gamma"] = i
        refer_args_infos["gamma"] = {"accuracy": scores, "args": gamma_range.tolist()}
        print("\n  -> ", s_max, svm_args["C"])
        # plt.close()
        # plt.plot(gamma_range, scores)
        # plt.savefig("../Data/gamma.png")
        return svm_args, refer_args_infos


class SRTClassifyImage:
    """
    分类图像
    """

    def __init__(self, imd_file):
        """ 图像分类

        :param imd_file: 图像文件
        """
        self.imd = EnviFileIO(imd_file).readToArray(interleave="b,r,c")
        print(self.imd.shape)
        self.class_envi = EnviFileIO()
        self.class_envi.im_infos = envi_class_file_info
        for i in range(self.imd.shape[0]):
            self.imd[i] = np.clip(self.imd[i], re_hist[i, 0], re_hist[i, 1])
            self.imd[i] = (self.imd[i] - re_hist[i, 0]) / (re_hist[i, 1] - re_hist[i, 0])
        self.d_select = []
        self.n_jdt_all = self.imd.shape[1]
        self.n_jdt_duan1 = int(self.imd.shape[1] / 5)
        self.n_jdt_duan2 = int(self.imd.shape[1] / 50)
        self.imdc = np.zeros((self.imd.shape[1], self.imd.shape[2]))

    def classify(self, mod, feat_index, save_image_file):
        """ 分类

        :param mod: 模型，带有predict接口
        :param feat_index: 特征的索引
        :param save_image_file: 保存的影像文件
        """
        self.d_select = [np.where(feat_names_all == c)[0][0] for c in feat_index]
        print("> Classify:", save_image_file)
        self.imdc = self.imdc * 0
        print("  ", end="")
        for i in range(self.imd.shape[1]):
            if i % self.n_jdt_duan1 == 0:
                print((i // self.n_jdt_duan1) * 20, end="")
            if i % self.n_jdt_duan2 == 0:
                print(".", end="")
            x = self.imd[self.d_select, i, :]
            self.imdc[i, :] = mod.predict(x.T)
        print("100")
        self.class_envi.saveToFile(self.imdc.astype("int8"), out_file=save_image_file)


def trainModels():
    save_info = {
        "spl_type": [],
        "feature_type": [],
        "class_algo_type": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "model_file_name": [],
        "imc_npy_file": [],
        "imc_geo_file": [],
        "confusion_matrix": [],
        "algo_info_file": []
    }

    save_dir = os.path.join(MODEL_DIR, TEMP_DIR)
    train_csv_fn = ""
    test_csv_fn = ""
    imdc_fn = ""

    category_names = ["IS", "VEG", "SOIL", "WATER"]

    if not os.path.isdir(save_dir):
        print(save_dir)
        os.mkdir(save_dir)
    json_file_name = os.path.join(save_dir, "mod.json")

    srt_sample = SRTSample(train_csv_fn, test_csv_fn)  # 样本
    srt_feature = SRTFeature()  # 特征
    srt_class_algo = SRTClassAlgos()  # 算法
    srt_classify_image = SRTClassifyImage(imdc_fn)

    spl_types = ["SPLS", "NOSPLS"]
    class_algo_types = ["SVM", "RF"]

    run_time = RumTime(511 * len(spl_types) * len(class_algo_types))
    run_time.strat()
    ii = 0

    for spl_type in spl_types:
        for class_algo_type in class_algo_types:
            srt_feature.clear()

            while True:
                feat_index = srt_feature.get()

                if feat_index is None:
                    break

                ii += 1

                feat_type = "_".join(srt_feature.feat_types)
                x0, y0, x_test0, y_test0 = srt_sample.get(spl_type, feat_index)
                to_f = os.path.join(save_dir, "_".join([spl_type, class_algo_type, feat_type]))

                save_info["spl_type"].append(spl_type)
                save_info["feature_type"].append(feat_type)
                save_info["class_algo_type"].append(class_algo_type)

                save_info["model_file_name"].append(to_f + "_mod.model")
                save_info["imc_npy_file"].append(to_f + "_d.npy")
                save_info["imc_geo_file"].append(to_f + "_imc.dat")
                save_info["algo_info_file"].append(to_f + "_ainfo.json")

                print(ii, "> ", " | ".join([spl_type, class_algo_type, feat_type]))

                # Training
                save_mod_fn = save_info["model_file_name"][-1]
                if class_algo_type == "RF":
                    algo_info = srt_class_algo.RF(x0.values, y0, x_test0.values, y_test0, save_mod_fn)
                elif class_algo_type == "SVM":
                    algo_info = srt_class_algo.SVM(x0.values, y0, x_test0.values, y_test0, save_mod_fn)
                else:
                    raise Exception("Can not find class type " + class_algo_type)

                save_info["train_accuracy"].append(srt_class_algo.train_acc * 100)
                save_info["test_accuracy"].append(srt_class_algo.test_acc * 100)
                save_cm_file = os.path.join(save_dir, "cm.txt")
                n = saveCM(srt_class_algo.cm, save_cm_file,
                           cate_names=category_names,
                           infos=[spl_type, class_algo_type, feat_type])
                save_info["confusion_matrix"].append(n)

                # Image classification
                srt_classify_image.classify(srt_class_algo.model, feat_index, save_info["imc_geo_file"][-1])
                algo_info["spl_type"] = spl_type
                algo_info["feature_type"] = feat_type
                algo_info["class_algo_type"] = class_algo_type
                with open(save_info["algo_info_file"][-1], "w", encoding="utf-8") as fs:
                    json.dump(algo_info, fs)

                print("* test_accuracy:{:>6.3f}".format(save_info["test_accuracy"][-1]))
                print("* train_accuracy:{:>6.3f}".format(save_info["train_accuracy"][-1]))

                run_time.add()
                run_time.printInfo()

                print()

    with open(json_file_name, 'w') as f:
        json.dump(save_info, f)
    df = pd.DataFrame(save_info)
    df = df.sort_values("test_accuracy", ascending=False)
    print(df)
    df.to_csv(os.path.join(save_dir, "info.csv"), index=False)

    return df


def main():
    fn0 = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910"
    with open(r"F:\ProjectSet\Shadow\QingDao\Mods\Temp\temp1.txt", "w", encoding="utf-8") as fw:
        for fn in os.listdir(fn0):
            if os.path.splitext(fn)[1] == ".dat":
                ff = os.path.join(fn0, fn)
                print(ff)


if __name__ == "__main__":
    fn0 = r"F:\ProjectSet\Shadow\QingDao\Mods\20230707H200910"
    print(fn0)
    with open(r"F:\ProjectSet\Shadow\QingDao\Mods\Temp\temp1.txt", "w", encoding="utf-8") as fw:
        for fn in os.listdir(fn0):
            if os.path.splitext(fn)[1] == ".dat":
                ff = os.path.join(fn0, fn)
                print(ff)
