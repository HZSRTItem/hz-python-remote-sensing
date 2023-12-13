# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowTraining.py
@Time    : 2023/7/4 8:59
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ShadowTraining
           https://zhuanlan.zhihu.com/p/642920484
-----------------------------------------------------------------------------"""
import itertools
import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import samplingToCSV
from SRTCodes.ModelTraining import CategoryTraining, ConfusionMatrix
from SRTCodes.NumpyUtils import saveCM, fmtCM
from SRTCodes.SRTSample import CSVSamples
from SRTCodes.Utils import RumTime, printList
from Shadow.ShadowImdC import ShadowImageClassification


def trainRF(d_train, labels):
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
    clf = RandomForestClassifier(
        n_estimators=rf_args["n_estimators"]
        , max_depth=rf_args["max_depth"]
        , min_samples_leaf=rf_args["min_samples_leaf"]
        , min_samples_split=rf_args["min_samples_split"]
        , n_jobs=-1, random_state=90)
    clf.fit(d_train, labels)
    ret_d = {"type": "RF", "args": rf_args, "train": refer_args_infos}
    return clf, ret_d


def trainSvm(d_train, labels):
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
    clf = SVC(
        kernel=svm_args["kernel"],
        C=svm_args["C"],
        gamma=svm_args["gamma"],
        cache_size=5000)
    clf.fit(d_train, labels)
    ret_d = {"type": "SVM", "args": svm_args, "train": refer_args_infos}
    return clf, ret_d


def trainRF_nocv(d_train, labels):
    rf_args = {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 1, "min_samples_split": 18}
    refer_args_infos = {}
    # Tuning parameters: n_estimators -----------------------------------------
    print("n_estimators: ", end="")
    scorel, s_max, canshu = [], 0, list(range(1, 150, 10))
    for i in canshu:
        rfc = RandomForestClassifier(n_estimators=i, n_jobs=-1, random_state=90)
        rfc.fit(d_train, labels)
        score = rfc.score(d_train, labels)
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
        rfc.fit(d_train, labels)
        score = rfc.score(d_train, labels)
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
        rfc.fit(d_train, labels)
        score = rfc.score(d_train, labels)
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
        rfc.fit(d_train, labels)
        score = rfc.score(d_train, labels)
        print(f"{i}:{score * 100:.2f}", end=" ")
        scorel.append(score)
        if score > s_max:
            s_max = score
            rf_args["min_samples_split"] = i
    print("\n  -> ", max(scorel), rf_args["min_samples_split"])
    refer_args_infos["min_samples_split"] = {"accuracy": scorel, "args": canshu}
    clf = RandomForestClassifier(
        n_estimators=rf_args["n_estimators"]
        , max_depth=rf_args["max_depth"]
        , min_samples_leaf=rf_args["min_samples_leaf"]
        , min_samples_split=rf_args["min_samples_split"]
        , n_jobs=-1, random_state=90)
    clf.fit(d_train, labels)
    ret_d = {"type": "RF", "args": rf_args, "train": refer_args_infos}
    return clf, ret_d


def trainSvm_nocv(d_train, labels):
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
        clf.fit(d_train, labels)
        score = clf.score(d_train, labels)
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
        clf.fit(d_train, labels)
        score = clf.score(d_train, labels)
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
    clf = SVC(
        kernel=svm_args["kernel"],
        C=svm_args["C"],
        gamma=svm_args["gamma"],
        cache_size=5000)
    clf.fit(d_train, labels)
    ret_d = {"type": "SVM", "args": svm_args, "train": refer_args_infos}
    return clf, ret_d


class FrontShadowCategoryTraining(CategoryTraining):

    def __init__(self, model_dir, model_name="model", n_category=2, category_names=None):
        super().__init__(model_dir=model_dir, model_name=model_name, n_category=n_category,
                         category_names=category_names)

        self._feat_iter = []
        self._n_iter = 0
        self._feat_types = ()
        self._feats = []

        self.mod_args = None
        self.csv_spl: CSVSamples = None
        self.test_x = None
        self.test_y = None

        self.tag_types = {}
        self.feat_types = {}
        self.spl_types = {}
        self.mod_types = {}

        self.front_feat_types = []
        self.front_feat_type_name = ""
        self.test_field_name = "TEST"

        self.sic: ShadowImageClassification = None

    def addSampleType(self, spl_type_name: str, *categorys):
        self._addCheck("c_name", categorys)
        self.spl_types[spl_type_name] = list(categorys)

    def addFeatureType(self, feat_type_name: str, *features):
        self._addCheck("feat_name", features)
        self.feat_types[feat_type_name] = list(features)

    def addTagType(self, tag_type_name: str, *tags):
        self._addCheck("tag_name", tags)
        self.tag_types[tag_type_name] = list(tags)

    def addModelType(self, mod_type_name: str, model_train_func):
        self.mod_types[mod_type_name] = model_train_func

    def addFrontFeatType(self, front_feat_type_name, *features):
        self._addCheck("feat_name", features)
        self.front_feat_type_name = front_feat_type_name
        self.front_feat_types = list(features)

    def _addCheck(self, name, values):
        for k in values:
            if not self.csv_spl.isIn(name, k):
                raise Exception("{0} not in {1}.".format(k, name))

    def saveDataToModelDirectory(self):
        self.csv_spl.saveToFile(os.path.join(self.model_dir, "train_data.csv"))

    def _initFeatures(self):
        feat_names = list(self.feat_types.keys())
        if self.front_feat_type_name != "":
            self._feat_iter.append(())
        for i in range(len(feat_names)):
            self._feat_iter.extend(itertools.combinations(feat_names, i + 1))

    def saveModArgs(self, fn):
        with open(fn, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.mod_args))
            # json.dump(f, self.mod_args)


class ShadowCategoryTraining(FrontShadowCategoryTraining):

    def __init__(self, model_dir, model_name, n_category=2, category_names=None):
        super(ShadowCategoryTraining, self).__init__(model_dir, model_name=model_name,
                                                     n_category=n_category, category_names=category_names)

        self.save_cm_file = None
        self.logAddField()
        self.timeModelDir()
        ...

    def logAddField(self):
        self._log.addField("MOD_TYPE", "string")
        self._log.addField("SPL_TYPE", "string")
        self._log.addField("FEAT_TYPE", "string")
        self._log.addField("TAG_TYPE", "string")
        self._log.addField("TRAIN_CM", "int")
        self._log.addField("TEST_CM", "int")
        self._log.addField("NUMBER", "int")
        self._log.printOptions(
            print_type="keyword",
            print_float_decimal=3,
            print_sep="\n",
            print_field_names=[
                "OATrain", "KappaTrain", "OATest", "KappaTest",
                "MOD_TYPE", "SPL_TYPE", "FEAT_TYPE", "TAG_TYPE"])

    def setSample(self, spl: CSVSamples = None):
        if spl is not None:
            self.csv_spl = spl
        self.test_x, self.test_y = self.csv_spl.get()
        select = self.test_x[self.test_field_name].values == 0
        self.test_x = self.test_x[select]
        self.test_y = self.test_y[select]
        y1 = self.test_y[self.test_y >= 5]
        self.test_y[self.test_y >= 5] = y1 - 4

    def addLog(self, d: dict):
        for k in d:
            self._log.addField(k, "int")

    def train(self, *args, **kwargs):
        # self.timeModelDir()
        self.saveDataToModelDirectory()
        self.save_cm_file = os.path.join(self.model_dir, "cm.txt")

        if self.front_feat_type_name != "":
            self.addLog({self.front_feat_type_name: self.front_feat_types})

        self.addLog(self.mod_types)
        self.addLog(self.spl_types)
        self.addLog(self.feat_types)
        self.addLog(self.tag_types)

        self._log.newLine()
        self._log.printFirstLine(is_to_file=True)
        self._log.saveHeader()
        self._initFeatures()

        self.run_time = RumTime(len(self.spl_types) * len(self._feat_iter) * len(self.tag_types) * len(self.mod_types))
        self.run_time.strat()

        n_run = 1
        print(self.model_dir)
        for tag_type in self.tag_types:
            tags = self.tag_types[tag_type]
            for mod_type in self.mod_types:
                mod_train = self.mod_types[mod_type]
                for spl_type in self.spl_types:
                    spls = self.spl_types[spl_type]

                    while self.getFeature():

                        feat_types, feats = self._feat_types, self._feats

                        mod_names = [spl_type, mod_type, tag_type] + list(feat_types)
                        mod_name = "-".join(mod_names)
                        mod_fn = os.path.join(self.model_dir, mod_name + "_mod.model")
                        mod_args_fn = os.path.join(self.model_dir, mod_name + "_args.json")

                        print("{0}. {1}".format(n_run, " ".join(mod_names)))

                        # train running ---
                        x_train, y_train, x_test, y_test = self.getSample(spls, feats, tags)
                        mod, mod_args = mod_train(x_train, y_train)

                        self.model = mod
                        self.saveModel(mod_fn)
                        self.mod_args = mod_args
                        self.mod_args["model_name"] = mod_name
                        self.mod_args["model_filename"] = mod_fn
                        self.mod_args["features"] = feats.copy()
                        self.mod_args["number"] = n_run
                        self.saveModArgs(mod_args_fn)

                        y_train_2 = self.model.predict(x_train)
                        self.train_cm.addData(y_train, y_train_2)
                        self.updateLogTrainCM()
                        y_test_2 = self.model.predict(x_test)
                        self.test_cm.addData(y_test, y_test_2)
                        self.updateLogTestCM()

                        train_cm_arr = self.train_cm.calCM()
                        n = saveCM(train_cm_arr, self.save_cm_file, cate_names=self.category_names,
                                   infos=["TRAIN"] + mod_names)
                        self._log["TRAIN_CM"] = n
                        test_cm_arr = self.test_cm.calCM()
                        n = saveCM(test_cm_arr, self.save_cm_file, cate_names=self.category_names,
                                   infos=["TEST"] + mod_names)
                        self._log["TEST_CM"] = n

                        self._log["ModelName"] = mod_name
                        self._log["MOD_TYPE"] = mod_type
                        self._log["SPL_TYPE"] = spl_type
                        self._log["FEAT_TYPE"] = "-".join(feat_types)
                        self._log["TAG_TYPE"] = tag_type
                        self._log["NUMBER"] = n_run

                        self._log[spl_type] = 1
                        for feat_type in feat_types:
                            self._log[feat_type] = 1
                        self._log[mod_type] = 1
                        self._log[tag_type] = 1

                        self._log.saveLine()
                        self._log.print(is_to_file=True)
                        self._log.newLine()

                        self.run_time.add()
                        self.run_time.printInfo()
                        n_run += 1
                        print()

                        self.train_cm.clear()
                        self.test_cm.clear()

                        self.sic.classify(mod_fn, feats, mod_name)

    def getSample(self, spls: list, feats: list, tags: list):
        x_test = self.test_x[feats].values
        y_test = self.test_y

        feats.append(self.test_field_name)
        x, y = self.csv_spl.get(c_names=spls, feat_names=feats, tags=tags)
        select = x[self.test_field_name].values
        x = x.values
        x_train, y_train = x[select == 1], y[select == 1]
        x_train = x_train[:, :-1]
        y1 = y_train[y_train >= 5]
        y_train[y_train >= 5] = y1 - 4
        return x_train, y_train, x_test, y_test

    def saveModel(self, model_name, *args, **kwargs):
        if model_name is not None:
            joblib.dump(self.model, model_name)

    def getFeature(self):
        if self._n_iter == len(self._feat_iter):
            self._n_iter = 0
            return False
        if self.front_feat_type_name != "":
            self._feat_types = [self.front_feat_type_name] + list(self._feat_iter[self._n_iter])
        else:
            self._feat_types = list(self._feat_iter[self._n_iter])
        self._feats = self.front_feat_types.copy()
        for feat_type in self._feat_types:
            if feat_type == self.front_feat_type_name:
                continue
            self._feats.extend(self.feat_types[feat_type])
        self._n_iter += 1
        return True

    def print(self):
        super(ShadowCategoryTraining, self).print()
        printList("Train Log Fields:", [field for field in self._log])
        self.csv_spl.print()
        print()
        printList("- Model Types: ", [k for k in self.mod_types])
        print()
        for k in self.spl_types:
            printList("+ Sample Types [{0}]:".format(k), self.spl_types[k])
        print()
        for k in self.feat_types:
            printList("* Feature Types [{0}]:".format(k), self.feat_types[k])
        print()
        for k in self.tag_types:
            printList("> Tag Types [{0}]:".format(k), self.tag_types[k])
        print()

    def initSIC(self, dat_fn):
        self.sic = ShadowImageClassification(dat_fn, self.model_dir)
        self.sic.is_trans = True
        self.sic.is_scale_01 = True

    def initCSVSample(self, spl_fn, cnames: list, cname="CNAME", tag="TAG"):
        self.csv_spl = CSVSamples(spl_fn)
        self.csv_spl.fieldNameCategory(cname)
        self.csv_spl.fieldNameTag(tag)
        self.csv_spl.addCategoryNames(cnames)
        self.csv_spl.readData()

    def sicAddCategory(self, name: str, color: tuple = None):
        self.sic.addCategory(name, color)

    def featureCallBack(self, feat_name, callback_func, is_trans=None):
        self.csv_spl.featureCallBack(feat_name=feat_name, callback_func=callback_func, is_trans=is_trans)
        self.sic.featureCallBack(feat_name=feat_name, callback_func=callback_func, is_trans=is_trans)

    def featureScaleMinMax(self, feat_name, x_min, x_max, is_trans=None, is_01=None):
        self.csv_spl.featureScaleMinMax(feat_name=feat_name, x_min=x_min, x_max=x_max, is_trans=is_trans, is_01=is_01)
        self.sic.featureScaleMinMax(feat_name=feat_name, x_min=x_min, x_max=x_max, is_trans=is_trans, is_01=is_01)


class ShadowCategoryTrainImdcOne:
    GRS = {}

    def __init__(self, model_dir, mod_name="model_name"):
        self.sic: ShadowImageClassification = None

        self.test_y = None
        self.test_x = None
        self.test_field_name = "TEST"

        self.csv_spl = CSVSamples()
        self.gr = GDALRaster()
        self.mod_name = mod_name

        self.feats = []
        self.categorys = []
        self.tags = []

        self.cm_names = []

        self.train_func = trainRF

        self.model_dir = model_dir
        dirname = self.timeModelDir()
        self.save_cm_file = os.path.join(self.model_dir, dirname + "_cm.txt")
        self.save_train_spl_file = os.path.join(self.model_dir, dirname + "_train_spl.txt")
        self.imd = None

    def setSample(self, spl: CSVSamples = None, is_sh_to_no=True):
        if spl is not None:
            self.csv_spl = spl
        self.test_x, self.test_y = self._getTrainTestSample(0)
        if is_sh_to_no:
            self.test_y = self._shCodeToNO(self.test_y)
        self._getNames()
        self.csv_spl.saveToFile(self.save_train_spl_file)

    def _getTrainTestSample(self, select_code, c_names=None, feat_names=None, tags=None):
        x, y = self.csv_spl.get(c_names=c_names, feat_names=feat_names, tags=tags)
        select = x[self.test_field_name].values == select_code
        x = x[select]
        y = y[select]
        return x, y

    def _getNames(self):
        self.feats = self.csv_spl.getFeatureNames()
        self.categorys = self.csv_spl.getCategoryNames()
        self.tags = self.csv_spl.getTagNames()

    def addGDALRaster(self, raster_fn, ):
        raster_fn = os.path.abspath(raster_fn)
        if raster_fn not in self.GRS:
            self.GRS[raster_fn] = GDALRaster(raster_fn)
        self.gr = self.GRS[raster_fn]
        return self.gr

    def fitFeatureNames(self, *feat_names):
        self.feats = list(feat_names)

    def fitCategoryNames(self, *c_names):
        self.categorys = list(c_names)

    def fitTagNames(self, *tag_names):
        self.tags = list(tag_names)

    def fitCMNames(self, *cm_names):
        self.cm_names = list(cm_names)

    def trainFunc(self, train_func):
        self.train_func = train_func

    def trainRF(self):
        self.train_func = trainRF

    def trainSvm(self):
        self.train_func = trainSvm

    def getSample(self, spls: list, feats: list, tags: list, is_sh_to_no=True):
        x_test = self.test_x[feats].values
        y_test = self.test_y
        feats.append(self.test_field_name)
        x_train, y_train = self._getTrainTestSample(1, c_names=spls, feat_names=feats, tags=tags)
        x_train = x_train.values[:, :-1]
        if is_sh_to_no:
            y_train = self._shCodeToNO(y_train)
        return x_train, y_train, x_test, y_test

    def _shCodeToNO(self, y_train):
        y1 = y_train[y_train >= 5]
        y_train[y_train >= 5] = y1 - 4
        return y_train

    def timeModelDir(self):
        dir_name = time.strftime("%Y%m%dH%H%M%S")
        self.model_dir = os.path.join(self.model_dir, dir_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        return dir_name

    def saveModel(self, model_name, *args, **kwargs):
        if model_name is not None:
            joblib.dump(self.model, model_name)

    def saveModArgs(self, fn):
        with open(fn, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.mod_args))
            # json.dump(f, self.mod_args)

    def initSIC(self, dat_fn=None):
        if dat_fn is None:
            dat_fn = self.gr.gdal_raster_fn
        self.sic = ShadowImageClassification(dat_fn, self.model_dir)
        self.sic.is_trans = True
        self.sic.is_scale_01 = True

    def sicAddCategory(self, name: str, color: tuple = None):
        self.sic.addCategory(name, color)

    def featureCallBack(self, feat_name, callback_func, is_trans=None):
        self.csv_spl.featureCallBack(feat_name=feat_name, callback_func=callback_func, is_trans=is_trans)
        self.sic.featureCallBack(feat_name=feat_name, callback_func=callback_func, is_trans=is_trans)

    def featureScaleMinMax(self, feat_name, x_min, x_max, is_trans=None, is_01=None):
        self.csv_spl.featureScaleMinMax(feat_name=feat_name, x_min=x_min, x_max=x_max, is_trans=is_trans, is_01=is_01)
        self.sic.featureScaleMinMax(feat_name=feat_name, x_min=x_min, x_max=x_max, is_trans=is_trans, is_01=is_01)

    def fit(self, is_sh_to_no=True):
        mod_name = self.mod_name
        mod_fn = os.path.join(self.model_dir, mod_name + "_mod.model")
        mod_args_fn = os.path.join(self.model_dir, mod_name + "_args.json")

        # train running ---
        print(">>> Train")
        printList("categorys", self.categorys)
        printList("feats", self.feats)
        printList("tags", self.tags)
        x_train, y_train, x_test, y_test = self.getSample(self.categorys, self.feats, self.tags, is_sh_to_no)
        mod, mod_args = self.train_func(x_train, y_train)

        # Save
        self.model = mod
        self.saveModel(mod_fn)
        self.mod_args = mod_args
        self.mod_args["model_name"] = mod_name
        self.mod_args["model_filename"] = mod_fn
        self.mod_args["features"] = self.feats.copy()
        self.mod_args["categorys"] = self.categorys.copy()
        self.mod_args["tags"] = self.tags.copy()
        self.saveModArgs(mod_args_fn)

        # Confusion Matrix
        print(">>> Confusion Matrix")
        cm = ConfusionMatrix(len(self.cm_names), self.cm_names)
        # Train
        y_train_2 = self.model.predict(x_train)
        cm.addData(y_train, y_train_2)
        cm_arr = cm.calCM()
        n = saveCM(cm_arr, self.save_cm_file, cate_names=self.cm_names, infos=["TRAIN"])
        print("* Train")
        print(fmtCM(cm_arr, cate_names=self.cm_names))
        cm.clear()
        # Test
        y_test_2 = self.model.predict(x_test)
        cm.addData(y_test, y_test_2)
        cm_arr = cm.calCM()
        n = saveCM(cm_arr, self.save_cm_file, cate_names=self.cm_names, infos=["TEST"])
        print("* Test")
        print(fmtCM(cm_arr, cate_names=self.cm_names))

        # Class
        self.sic.classify(mod_fn, self.feats, mod_name)

    def addCSVFile(self, spl_fn, is_spl=False):
        if is_spl:
            samplingToCSV(spl_fn, self.gr, spl_fn)
        self.csv_spl = CSVSamples(spl_fn)
