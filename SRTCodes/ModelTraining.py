# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ModelTraining.py
@Time    : 2023/6/22 16:08
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of ModelTraining
-----------------------------------------------------------------------------"""
import csv
import os
import time

import numpy as np

from SRTCodes.SRTCollection import SRTCollection, SRTCollectionDict
from SRTCodes.Utils import Jdt

eps = 0.000001


class TrainLog(SRTCollection):

    def __init__(self, save_csv_file="train_save.csv", log_filename="train_log.txt"):
        super(TrainLog, self).__init__()

        self.field_names = []
        self.field_types = []
        self.field_index = {}
        self.field_datas = []
        self.n_datas = 0

        self.is_save_log = False
        self.log_filename = log_filename
        # "trainlog" + time.strftime("%Y%m%d") + ".txt"

        self.print_type = "table"  # table or keyword
        self.print_type_fmts = {"table": "{0}", "keyword": "{1}: {0}"}
        self.print_sep = "\t"
        self.print_field_names = []
        self.print_column_fmt = []
        self.print_float_decimal = 3
        self.print_type_column_width = {"int": 6, "float": 12, "string": 26}
        self.print_type_init_v = {"int": 0, "float": 0.0, "string": ""}
        self.print_type_column_fmt = {}
        self._getTypeColumnFmt()

        self.save_csv_file = save_csv_file

    def toDict(self):
        to_dict = {
            "_n_iter": self._n_iter,
            "_n_next": self._n_next,
            "field_names": self.field_names,
            "field_types": self.field_types,
            "field_index": self.field_index,
            "field_datas": self.field_datas,
            "n_datas": self.n_datas,
            "is_save_log": self.is_save_log,
            "log_filename": self.log_filename,
            "print_type": self.print_type,
            "print_type_fmts": self.print_type_fmts,
            "print_sep": self.print_sep,
            "print_field_names": self.print_field_names,
            "print_column_fmt": self.print_column_fmt,
            "print_float_decimal": self.print_float_decimal,
            "print_type_column_width": self.print_type_column_width,
            "print_type_init_v": self.print_type_init_v,
            "print_type_column_fmt": self.print_type_column_fmt,
            "save_csv_file": self.save_csv_file,
        }
        return to_dict

    def loadDict(self, to_dict):
        self._n_iter = to_dict["_n_iter"]
        self._n_next = to_dict["_n_next"]
        self.field_names = to_dict["field_names"]
        self.field_types = to_dict["field_types"]
        self.field_index = to_dict["field_index"]
        self.field_datas = to_dict["field_datas"]
        self.n_datas = to_dict["n_datas"]
        self.is_save_log = to_dict["is_save_log"]
        self.log_filename = to_dict["log_filename"]
        self.print_type = to_dict["print_type"]
        self.print_type_fmts = to_dict["print_type_fmts"]
        self.print_sep = to_dict["print_sep"]
        self.print_field_names = to_dict["print_field_names"]
        self.print_column_fmt = to_dict["print_column_fmt"]
        self.print_float_decimal = to_dict["print_float_decimal"]
        self.print_type_column_width = to_dict["print_type_column_width"]
        self.print_type_init_v = to_dict["print_type_init_v"]
        self.print_type_column_fmt = to_dict["print_type_column_fmt"]
        self.save_csv_file = to_dict["save_csv_file"]

    def _getTypeColumnFmt(self):
        self.print_type_column_fmt = {
            "int": "{:>" + str(self.print_type_column_width["int"]) + "d}",
            "float": "{:>" + str(self.print_type_column_width["float"]) + "." + str(self.print_float_decimal) + "f}",
            "string": "{:>" + str(self.print_type_column_width["string"]) + "}"
        }

    def addField(self, field_name, field_type="string", init_v=None):
        if field_name in self.field_names:
            raise Exception("Error: field name \"{0}\" have in field names.".format(field_name))
        self.field_names.append(field_name)
        self.field_types.append(field_type)
        if init_v is not None:
            self.print_type_init_v[field_type] = init_v
        self.field_index[field_name] = len(self.field_names) - 1
        self._n_next.append(field_name)

    def getFieldName(self, idx):
        return self.field_names[idx]

    def getFieldIndex(self, name):
        return self.field_index[name]

    def printOptions(self, print_type="table", print_sep="\t", print_field_names=None, print_column_fmt=None,
                     print_float_decimal=3):
        """ print options
        /
        :param print_float_decimal: float decimal
        :param print_type: table or keyword default "table"
        :param print_sep: sep default "\t"
        :param print_field_names: default all
        :param print_column_fmt: default get from field type
        :return: None
        """
        self.print_type = print_type  # table or keyword
        self.print_sep = print_sep
        self.print_float_decimal = print_float_decimal

        if print_field_names is None:
            print_field_names = []
        if not print_field_names:
            print_field_names = self.field_names.copy()
        self.print_field_names = print_field_names

        if print_column_fmt is None:
            print_column_fmt = []
        if not print_column_fmt:
            for name in self.print_field_names:
                ft = self.field_types[self.getFieldIndex(name)]
                print_column_fmt.append(self.print_type_column_fmt[ft])
        self.print_column_fmt = print_column_fmt

    def print(self, front_str=None, is_to_file=False, end="\n", line_idx=-1):
        if front_str is not None:
            print(front_str, end="")
        lines = self.field_datas[line_idx]
        self._printLine(lines, end=end)
        if is_to_file:
            with open(self.log_filename, "a", encoding="utf-8") as fw:
                self._printLine(lines, end=end, file=fw)

    def _printLine(self, lines, end="\n", file=None):
        for i, name in enumerate(self.print_field_names):
            d = self.print_column_fmt[i].format(lines[self.getFieldIndex(name)])
            d = d.strip()
            fmt_d = self.print_type_fmts[self.print_type].format(d, name)
            print(fmt_d, end=self.print_sep, file=file)
        if self.print_sep != "\n":
            print(end=end, file=file)

    def updateField(self, field_name, field_data, idx_field_data=-1, newline=False):
        if len(self.field_datas) == 0:
            newline = True
        if newline:
            self.field_datas.append(self._initFieldDataLine())
        self.field_datas[idx_field_data][self.getFieldIndex(field_name)] = field_data

    def newLine(self):
        self.field_datas.append(self._initFieldDataLine())

    def _initFieldDataLine(self):
        line = []
        for i in range(len(self.field_names)):
            line.append(self.print_type_init_v[self.field_types[i]])
        return line

    def printFirstLine(self, end="\n", is_to_file=False):
        for name in self.print_field_names:
            print(name, end=self.print_sep)
        print(end=end)
        if is_to_file:
            with open(self.log_filename, "a", encoding="utf-8") as fw:
                for name in self.print_field_names:
                    print(name, end=self.print_sep, file=fw)
                print(end=end, file=fw)

    def saveLine(self, n_line=-1):
        if self.save_csv_file is not None:
            with open(self.save_csv_file, "a", encoding="utf-8", newline="") as fw:
                cw = csv.writer(fw)
                cw.writerow(self.field_datas[n_line])

    def saveHeader(self):
        if self.save_csv_file is not None:
            with open(self.save_csv_file, "w", encoding="utf-8", newline="") as fw:
                cw = csv.writer(fw)
                cw.writerow(self.field_names)

    def __getitem__(self, field_name):
        field_name_idx = self.getFieldIndex(field_name)
        return self.field_datas[-1][field_name_idx]

    def __setitem__(self, field_name, value):
        field_name_idx = self.getFieldIndex(field_name)
        self.field_datas[-1][field_name_idx] = value


class ConfusionMatrix:

    def __init__(self, n_class=0, class_names=None):
        """
                预测
            |  0  |  1  |
             -----------    制
        真 0 |    |    |    图
        实 1 |    |    |    精
        ------------------  度
              用户精度

        :param n_class: number of category
        :param class_names: names of category
        """
        self._n_class = n_class
        self._class_names = class_names
        if n_class == 0:
            if class_names is not None:
                n_class = len(class_names)
            else:
                return
        self._cm = np.zeros((n_class, n_class))
        self._cm_accuracy = self.calCM()
        if self._class_names is None:
            self._class_names = ["CATEGORY_" + str(i + 1) for i in range(n_class)]
        elif len(self._class_names) != n_class:
            raise Exception("The number of category names is different from the number of input categories.")
        self._it_count = 0

    def toDict(self):
        to_dict = {
            "_n_class": self._n_class,
            "_class_names": self._class_names,
        }
        return to_dict

    def CNAMES(self):
        return self._class_names

    def addData(self, y_true, y_pred):
        for i in range(len(y_true)):
            if int(y_true[i]) > 0 or int(y_pred[i]) > 0:
                if (int(y_pred[i]) - 1) == 40:
                    continue
                if (int(y_true[i]) <= 0) or (int(y_pred[i]) <= 0):
                    continue
                if (int(y_true[i]) > self._n_class) or (int(y_pred[i]) > self._n_class):
                    continue
                self._cm[int(y_true[i]) - 1, int(y_pred[i]) - 1] += 1
        self._cm_accuracy = self.calCM()

    def UA(self, idx_name=None):
        acc = self._cm_accuracy[self._n_class + 1, :self._n_class]
        if idx_name is None:
            return acc
        elif isinstance(idx_name, int):
            return acc[idx_name - 1]
        elif isinstance(idx_name, str):
            return acc[self._class_names.index(idx_name)]

    def PA(self, idx_name=None):
        acc = self._cm_accuracy[:self._n_class, self._n_class + 1]
        if idx_name is None:
            return acc
        elif isinstance(idx_name, int):
            return acc[idx_name - 1]
        elif isinstance(idx_name, str):
            return acc[self._class_names.index(idx_name)]

    def OA(self):
        return self._cm_accuracy[-1, -1]

    def getKappa(self):
        pe = np.sum(np.sum(self._cm, axis=0) * np.sum(self._cm, axis=1)) / (self._cm.sum() * self._cm.sum() + eps)
        return (self.OA() / 100 - pe) / (1 - pe + eps)

    #
    # def getKappa(self):
    #     pe = np.sum(self.OA() * self.PA()) / (self._cm.sum() * self._cm.sum() + eps)
    #     return (self.OA() - pe) / (1 - pe + eps)

    def printCM(self):
        print(self.fmtCM())

    def clear(self):
        self._cm = np.zeros((self._n_class, self._n_class))
        self._cm_accuracy = self.calCM()

    def fmtCM(self, cm: np.array = None, cate_names=None):
        if cm is None:
            cm = self._cm_accuracy
        if cate_names is None:
            cate_names = self._class_names
        fmt_row0 = "{:>8}"
        fmt_column0 = "{:>8}"
        fmt_number = "{:>8d}"
        fmt_float = "{:>8.2f}"
        n_cate = len(cate_names)
        out_s = ""
        out_s += fmt_column0.format("CM")
        for i in range(n_cate):
            out_s += " " + fmt_row0.format(cate_names[i])
        out_s += " " + fmt_row0.format("SUM")
        out_s += " " + fmt_row0.format("PA") + "\n"
        for i in range(n_cate):
            out_s += fmt_column0.format(cate_names[i])
            for j in range(n_cate):
                out_s += " " + fmt_number.format(int(cm[i, j]))
            out_s += " " + fmt_number.format(int(cm[i, n_cate]))
            out_s += " " + fmt_float.format(cm[i, n_cate + 1]) + "\n"
        out_s += fmt_column0.format("SUM")
        for i in range(n_cate):
            out_s += " " + fmt_number.format(int(cm[n_cate, i]))
        out_s += " " + fmt_number.format(int(cm[n_cate, n_cate]))
        out_s += " " + fmt_float.format(cm[n_cate, n_cate + 1]) + "\n"
        out_s += fmt_column0.format("UA")
        for i in range(n_cate):
            out_s += " " + fmt_float.format(cm[n_cate + 1, i])
        out_s += " " + fmt_float.format(cm[n_cate + 1, n_cate])
        out_s += " " + fmt_float.format(cm[n_cate + 1, n_cate + 1]) + "\n"
        return out_s

    def calCM(self, cm: np.array = None):
        if cm is None:
            cm = self._cm
        n_class = cm.shape[0]
        out_cm = np.zeros([n_class + 2, n_class + 2])
        out_cm[:n_class, :n_class] = cm
        out_cm[n_class, :] = np.sum(out_cm, axis=0)
        out_cm[:, n_class] = np.sum(out_cm, axis=1)
        out_cm[n_class + 1, :] = np.diag(out_cm) * 1.0 / (out_cm[n_class, :] + 0.00001) * 100
        out_cm[:, n_class + 1] = np.diag(out_cm) * 1.0 / (out_cm[:, n_class] + 0.00001) * 100
        out_cm[n_class + 1, n_class + 1] = (np.sum(np.diag(cm))) / (out_cm[n_class, n_class] + 0.00001) * 100
        return out_cm

    def categoryNames(self):
        return self._class_names

    def accuracy(self):
        """ OA PA UA """
        cm = self._cm_accuracy
        cate_names = self._class_names
        pa_d = self.PA().tolist()
        ua_d = self.UA().tolist()
        oa_d = np.diag(self._cm_accuracy)
        # oa_d = oa_d[:len(cate_names)] / (oa_d[-2] + 0.00001)
        # oa_d = oa_d.tolist()
        to_dict = {}
        for i, cate_name in enumerate(cate_names):
            to_dict[cate_name] = [float(oa_d[-1]), float(pa_d[i]), float(ua_d[i])]
        return to_dict

    def accuracyCategory(self, category):
        cname = category
        if isinstance(category, str):
            category = self._class_names.index(category)
        else:
            category = category - 1

        cm_category = np.zeros((2,2))
        cm_category[0, 0] = int(self._cm[category, category])
        cm_category[0, 1] = int(np.sum(np.diag(self._cm[category, :]))) -int(self._cm[category, category])
        cm_category[1, 0] = int(np.sum(np.diag(self._cm[:, category]))) -int(self._cm[category, category])
        cm_category[1, 1] = int(np.sum(self._cm)) - int(np.sum(cm_category))

        cm = ConfusionMatrix(2, [str(cname), "NOT_KNOW"])
        cm._cm = cm_category
        cm._cm_accuracy = cm.calCM()
        return cm


    def __iter__(self):
        return self

    def __next__(self):
        # 获取下一个数
        if self._it_count < len(self._class_names):
            result = self._class_names[self._it_count]
            self._it_count += 1
            return result
        else:
            self._it_count = 0
            raise StopIteration

    def toList(self):
        return self._cm.tolist()


class MeanSquareError:

    def __init__(self):
        self._n_spl = 0
        self.mse = 0

    def add(self, y_true, y_pred):
        err = (y_true - y_pred) ** 2
        mse = (self.mse * self._n_spl + np.sum(err)) / (self._n_spl + len(err))
        self.mse = mse
        self._n_spl += len(err)

    def MSE(self):
        return np.sqrt(self.mse)

    def clear(self):
        self._n_spl = 0
        self.mse = 0


class ConfusionMatrixCollection(SRTCollectionDict):

    def __init__(self, n_class=0, class_names=None):
        super().__init__()
        self.n_class = n_class
        self.class_names = class_names

    def toDict(self):
        to_dict = {
            "cms": {cm:self.n_next[cm].toDict() for cm in self.n_next}
        }
        return to_dict

    def addCM(self, name, n_class=0, class_names=None, cm: ConfusionMatrix = None) -> ConfusionMatrix:
        if cm is not None:
            self.n_next[name] = cm
            return self.n_next[name]
        if (n_class == 0) and (class_names is None):
            n_class = self.n_class
            class_names = self.class_names
        self.n_next[name] = ConfusionMatrix(n_class=n_class, class_names=class_names)
        return self.n_next[name]

    def __getitem__(self, item) -> ConfusionMatrix:
        return self.n_next[item]


class ConfusionMatrixLog:

    def __init__(self, n_category=2, category_names=None):
        self.cms = ConfusionMatrixCollection(n_class=n_category, class_names=category_names)
        self.log = None

    def toDict(self):
        to_dict = {
            "cms": self.cms.toDict(),
            "log": self.log.toDict() if self.log is not None else None,
        }
        return to_dict

    def addCM(self, name, cm=None, ):
        self.cms.addCM(name, cm=cm)

    def initLog(self, log_type: str, cm: ConfusionMatrix = None, log: TrainLog = None, ):
        cm, log = self.initlogcm(cm, log, log_type)
        log.addField("OA{}".format(log_type), "float")
        log.addField("Kappa{}".format(log_type), "float")
        for name in cm:
            log.addField(name + " UA{}".format(log_type), "float")
            log.addField(name + " PA{}".format(log_type), "float")

    def initlogcm(self, cm, log, log_type):
        if log is None:
            log = self.log
        else:
            self.log = log
        if cm is None:
            cm = self.cms[log_type]
        return cm, log

    def initThisLogs(self, log: TrainLog = None, ):
        for name in self.cms:
            self.initLog(name, cm=self.cms[name], log=log)

    def updateLog(self, log_type: str, cm: ConfusionMatrix = None, log: TrainLog = None, ):
        cm, log = self.initlogcm(cm, log, log_type)
        log.updateField("OA{}".format(log_type), cm.OA())
        log.updateField("Kappa{}".format(log_type), cm.getKappa())
        for name in cm:
            log.updateField(name + " UA{}".format(log_type), cm.UA(name))
            log.updateField(name + " PA{}".format(log_type), cm.PA(name))


class TrainTestConfusionMatrixLog(ConfusionMatrixLog):

    def __init__(self, n_category=2, category_names=None):
        super().__init__(n_category, category_names)
        self.train_cm = self.cms.addCM("Train")
        self.test_cm = self.cms.addCM("Test")
        self.log = None

    def initTrainLog(self, log: TrainLog = None):
        self.initLog("Train", log=log)

    def initTestLog(self, log: TrainLog = None):
        self.initLog("Test", log=log)

    def initTrainTestLog(self, log: TrainLog = None):
        self.initTrainLog(log=log)
        self.initTestLog(log=log)

    def updateLogTrainCM(self, log: TrainLog = None):
        self.updateLog("Train", log=log)

    def updateLogTestCM(self, log: TrainLog = None):
        self.updateLog("Test", log=log)

    def updateLogTrainTestCM(self, log: TrainLog = None):
        self.updateLogTrainCM(log=log)
        self.updateLogTestCM(log=log)

    def addTrainData(self, y_true, y_pred):
        self.train_cm.addData(y_true, y_pred)

    def addTestData(self, y_true, y_pred):
        self.test_cm.addData(y_true, y_pred)


class Training:

    def __init__(self, model_dir, model_name):
        if model_dir is None:
            return

        self.model_dir = model_dir
        self.model = None
        self.models = []
        self.model_name = model_name

        if self.model_dir is None:
            raise Exception("Model directory is None.")
        self.model_dir = os.path.abspath(self.model_dir)
        if not os.path.isdir(self.model_dir):
            raise Exception("Can not find model directory " + self.model_dir)

        self._log: TrainLog = None

    def toDict(self):
        to_dict = {
            "model_dir": self.model_dir,
            "model": str(self.model),
            "models": [str(model) for model in self.models],
            "model_name": self.model_name,
            "_log": self._log.toDict(),
        }
        return to_dict

    def _initLog(self):
        self._log = TrainLog(log_filename=os.path.join(self.model_dir, "train_log.txt"))
        self._log.addField("ModelName", "string")

    def addModel(self, model):
        self.model = model
        self.models.append(model)

    def train(self, *args, **kwargs):
        return self.model.train()

    def saveModel(self, model_name, *args, **kwargs):
        return self.model

    def timeModelDir(self):
        dir_name = time.strftime("%Y%m%dH%H%M%S")
        self.model_dir = os.path.join(self.model_dir, dir_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        self._log.save_csv_file = os.path.join(self.model_dir, "train_save_" + dir_name + ".csv")
        return dir_name

    def testAccuracy(self):
        return 0

    def print(self):
        print("Model", self.model)


class CategoryTraining(Training):

    def __init__(self, model_dir, model_name="model", n_category=2, category_names=None):
        """ Training init
        /
        :param n_category: number of category
        :param model_dir: save model directory
        :param category_names: category names
        """
        Training.__init__(self, model_dir, model_name)
        self._initLog()
        self.category_names = category_names

        self.train_cm = ConfusionMatrix(n_category, category_names)
        self._log.addField("OATrain", "float")
        self._log.addField("KappaTrain", "float")
        for name in self.train_cm:
            self._log.addField(name + " UATrain", "float")
            self._log.addField(name + " PATrain", "float")

        self.test_cm = ConfusionMatrix(n_category, category_names)
        self._log.addField("OATest", "float")
        self._log.addField("KappaTest", "float")
        for name in self.test_cm:
            self._log.addField(name + " UATest", "float")
            self._log.addField(name + " PATest", "float")

    def updateLogTrainCM(self):
        self._log.updateField("OATrain", self.train_cm.OA())
        self._log.updateField("KappaTrain", self.train_cm.getKappa())
        for name in self.train_cm:
            self._log.updateField(name + " UATrain", self.train_cm.UA(name))
            self._log.updateField(name + " PATrain", self.train_cm.PA(name))

    def updateLogTestCM(self):
        self._log.updateField("OATest", self.test_cm.OA())
        self._log.updateField("KappaTest", self.test_cm.getKappa())
        for name in self.test_cm:
            self._log.updateField(name + " UATest", self.test_cm.UA(name))
            self._log.updateField(name + " PATest", self.test_cm.PA(name))

    def testAccuracy(self):
        return self.test_cm.OA()


class RegressionTraining(Training):

    def __init__(self, model_dir, model_name="model"):
        Training.__init__(self, model_dir, model_name)
        self._initLog()

        self.train_mse = MeanSquareError()
        self._log.addField("MSETrain", "float")

        self.test_mse = MeanSquareError()
        self._log.addField("MSETest", "float")

        return

    def updateLogTrainMSE(self):
        self._log.updateField("MSETrain", self.train_mse.MSE())

    def updateLogTestMSE(self):
        self._log.updateField("MSETest", self.test_mse.MSE())

    def clearTrainMSE(self):
        self.train_mse.clear()

    def clearTestMSE(self):
        self.test_mse.clear()

    def testAccuracy(self):
        return self.test_mse.MSE()


def dataModelPredict(data, data_deal, is_jdt, model):
    if data_deal is None:
        data_deal = lambda _data: _data
    data_c = np.zeros((data.shape[1], data.shape[2]))
    jdt = Jdt(data.shape[1], "dataModelPredict").start(is_jdt=is_jdt)
    for i in range(data.shape[1]):
        jdt.add(is_jdt=is_jdt)
        x = data_deal(data[:, i, :].T)
        y = model.predict(x)
        data_c[i, :] = y
    jdt.end(is_jdt=is_jdt)
    return data_c


def dataPredictPatch(image_data, win_size, predict_func, is_jdt=True):
    imdc = np.zeros(image_data.shape[1:])
    win_row, win_column = win_size
    row_start, row_end, = win_row, imdc.shape[0] - win_row
    column_start, column_end, = win_column, imdc.shape[1] - win_column
    win_row_2, win_column_2 = int(win_row / 2), int(win_column / 2)
    row_01, column_01 = win_row%2, win_column%2
    col_imdc = np.zeros((column_end - column_start, image_data.shape[0], win_row, win_column))
    jdt = Jdt(row_end - row_start, "dataPredictPatch").start(is_jdt=is_jdt)
    for i in range(row_start, row_end):
        j_select = 0
        for j in range(column_start, column_end):
            r, c = i, j
            col_imdc[j_select] = image_data[:, r - win_row_2:r + win_row_2 + row_01, c - win_column_2:c + win_column_2 + column_01]
            j_select += 1
        y = predict_func(col_imdc)
        imdc[i, column_start: column_end] = y
        jdt.add(is_jdt=is_jdt)
    jdt.end(is_jdt=is_jdt)
    return imdc


class ModelDataCategory:

    def __init__(self, *datas):
        self.data_list = []
        self.addDatas(*datas)
        self.data = None

    def addData(self, data):
        self.data_list.append(data)

    def addDatas(self, *datas):
        self.data_list.extend(datas)

    def dataPredict(self, model, data_deal=None, is_jdt=False, *args, **kwargs):
        for data in self.data_list:
            dataModelPredict(data, data_deal, is_jdt, model)


def main():
    pass


if __name__ == "__main__":
    main()
