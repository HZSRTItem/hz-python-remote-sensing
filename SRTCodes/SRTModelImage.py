# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTModelImage.py
@Time    : 2024/3/27 21:13
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTModelTrainClass
-----------------------------------------------------------------------------"""
import os.path
import sys
from shutil import copyfile

import joblib
import numpy as np
import pandas as pd
import torch
from osgeo import gdal
from osgeo_utils.gdal_merge import main as gdal_merge_main
from sklearn.model_selection import train_test_split

from SRTCodes.GDALRasterClassification import GDALModelDataCategory
from SRTCodes.GDALRasterIO import GDALRasterChannel, saveGTIFFImdc, GDALRaster, tiffAddColorTable
from SRTCodes.ModelTraining import ConfusionMatrix, TrainLog, ConfusionMatrixLog, dataPredictPatch, dataModelPredict
from SRTCodes.NumpyUtils import categoryMap
from SRTCodes.PytorchModelTraining import PytorchTraining, pytorchModelCodeString
from SRTCodes.SRTFeature import SRTFeaturesMemory
from SRTCodes.Utils import Jdt, timeDirName, FN, numberfilename, changext, filterFileExt, changefiledirname, \
    funcCodeString, saveJson, datasCaiFen


def yToCM(y, y1, cm, category_names, map_dict):
    if category_names is None:
        category_names = np.unique(np.concatenate([np.unique(y), np.unique(y1)]))
        category_names = np.sort(category_names).tolist()
    if map_dict is None:
        cnames = np.unique(np.concatenate([np.unique(y), np.unique(y1)]))
        map_dict = {cnames[i]: i + 1 for i in range(len(cnames))}
    if cm is None:
        cm = ConfusionMatrix(class_names=category_names)
    y_tmp, y1_tmp = [], []
    for i in range(len(y)):
        y_tmp.append(map_dict[y[i]])
        y1_tmp.append(map_dict[y1[i]])
    cm.addData(y_tmp, y1_tmp)
    return cm


def funcPredict(model, x: torch.Tensor):
    logit = model(x)
    y = torch.argmax(logit, dim=1)
    return y


class SRTModelImageInit:

    def __init__(
            self,
            model_name=None,
            model_dirname=None,
            category_names=None,
            category_colors=None,
    ):
        self.mod_fn = None
        if category_names is None:
            category_names = ["NOT_KNOW"]

        self.model_name = model_name
        self.model_dirname = model_dirname
        self.category_names = category_names
        self.category_colors = category_colors

        self.ext_funcs = {}
        self.color_table = {}

        self.train_cm = ConfusionMatrix()
        self.test_cm = ConfusionMatrix()

    def initColorTable(self, color_table_dict, *code_colors, **kwargs):
        for c_code, color in color_table_dict.items():
            self.color_table[c_code] = color
        for c_code, color in code_colors:
            self.color_table[c_code] = color
        for c_code, color in kwargs.items():
            self.color_table[c_code] = color

    def addFeatExt(self, to_field_name, ext_func, *args, **kwargs):
        self.ext_funcs[to_field_name] = (ext_func, args, kwargs)

    def train(self, *args, **kwargs):
        return

    def imdc(self, *args, **kwargs):
        return

    def initImdc1(self, geo_fns, grc):
        if geo_fns is None:
            geo_fns = []
        if grc is None:
            grc = GDALRasterChannel()
        if geo_fns is not None:
            if isinstance(geo_fns, str):
                geo_fns = [geo_fns]
            for geo_fn in geo_fns:
                grc.addGDALDatas(geo_fn)
        if self.ext_funcs:
            for field_name, (ext_func, args, kwargs) in self.ext_funcs.items():
                grc.data[field_name] = ext_func(grc, *args, **kwargs)
        return grc

    def predict(self, *args, **kwargs):
        return []

    def saveCodeFile(self, *args, **kwargs, ):
        if ("code_fn" in kwargs) and ("to_code_fn" in kwargs):
            copyfile(kwargs["code_fn"], kwargs["to_code_fn"])

    def getToGeoFn(self, to_geo_fn):
        if to_geo_fn is None:
            if self.mod_fn is None:
                if len(sys.argv) >= 2:
                    self.mod_fn = sys.argv[1]
            if self.mod_fn is not None:
                to_geo_fn = changext(self.mod_fn, "_imdc.tif")
                to_geo_fn = numberfilename(to_geo_fn)
        if to_geo_fn is None:
            raise Exception("Can not get to geo filename.")
        return to_geo_fn


class SRTModImSklearn(SRTModelImageInit):

    def __init__(self):
        super().__init__()

        self.test_column = None
        self.clf = None
        self.df = pd.DataFrame()
        self.field_name_category = "CATEGORY"
        self.x_keys = []

        self.category_names = None

        self.x = None
        self.y = None

        self.x_train = None
        self.x_test = None
        self.y_test = None
        self.y_train = None

        self.select_list = None

    def initPandas(self, df: pd.DataFrame):
        self.df = df
        if "TEST" in self.df:
            self.test_column = self.df["TEST"].values

    def initCategoryField(self, field_name="CATEGORY", map_dict=None, is_notfind_to0=False,
                          is_string_code=False):
        if map_dict is None:
            map_dict = {}

        self.field_name_category = field_name

        y = self.df[self.field_name_category].values.tolist()

        if is_string_code:
            y_tmp = []
            curr_code = 1
            curr_code_dict = {}
            for k in y:
                if k not in curr_code_dict:
                    curr_code_dict[k] = curr_code
                    curr_code += 1
                y_tmp.append(curr_code_dict[k])
            y = y_tmp

        if map_dict is not None:
            y = categoryMap(y, map_dict=map_dict, is_notfind_to0=is_notfind_to0)

        self.y = np.array(y)
        return field_name

    def initXKeys(self, x_keys):
        self.x = self.df[x_keys]
        self.x_keys = x_keys

    def initCLF(self, clf):
        self.clf = clf

    def dfFilterEQ(self, field_name, data):
        self.df = self.df[self.df[field_name] == data]
        return self.df

    def filterCategory(self, *category):
        select_list = []
        for cate in self.y:
            cate = int(cate)
            select_list.append(cate in category)
        self.x = self.x[select_list]
        self.y = self.y[select_list]
        self.select_list = select_list
        return self.x, self.y

    def addFeatExt(self, to_field_name, ext_func, *args, **kwargs):
        super(SRTModImSklearn, self).addFeatExt(to_field_name, ext_func, *args, **kwargs)
        self.df[to_field_name] = ext_func(self.df, *args, **kwargs)

    def train(self, is_print=True, sample_weight=None, *args, **kwargs):
        x_test, x_train, y_test, y_train = self.getTrainingData()
        self.clf.fit(x_train.values, y_train, sample_weight=sample_weight)
        if is_print:
            print("train accuracy: {0}".format(self.clf.score(x_train.values, y_train)))
            if self.y_test is not None:
                print("test accuracy: {0}".format(self.clf.score(x_test.values, y_test)))

    def getTrainingData(self):
        if self.x_test is None:
            self.x_train, self.y_train = self.x, self.y
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test
        return x_test, x_train, y_test, y_train

    def imdc(self, to_geo_fn, geo_fns=None, grc: GDALRasterChannel = None, is_jdt=True, data_deal=None,
             is_print=True, ):
        """ c_code, color """
        grc = self.initImdc1(geo_fns, grc)
        gmdc = GDALModelDataCategory()
        if geo_fns is not None:
            gmdc.gr = GDALRaster(geo_fns[0])
        else:
            gmdc.gr = list(grc.GRS.values())[0]
        gmdc.addData(grc.fieldNamesToData(*self.x_keys))
        gmdc.fit(to_imdc_fns=[to_geo_fn], model=self.clf, is_jdt=is_jdt, data_deal=data_deal,
                 color_table=self.color_table)

    def imdcGeoFn(self, to_geo_fn, geo_fn, is_jdt=True, data_deal=None, ):
        gr = GDALRaster(geo_fn)
        gr.readAsArray()
        self.imdcGR(to_geo_fn, gr, data_deal, is_jdt)

    def imdcGR(self, to_geo_fn, gr, data_deal=None, is_jdt=True):
        imdc = dataModelPredict(gr.d, data_deal=data_deal, is_jdt=is_jdt, model=self.clf)
        gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
        tiffAddColorTable(to_geo_fn, 1, self.color_table)

    def imdcTiles(self, to_fn=None, tiles_dirname=None, tiles_fns=None, data_deal=None, is_jdt=True, ):
        to_fn = self.getToGeoFn(to_fn)
        if tiles_fns is None:
            tiles_fns = []
        tiles_fns = list(tiles_fns)
        if tiles_dirname is not None:
            tiles_fns.extend(filterFileExt(tiles_dirname, ext=".tif"))
        to_tiles_dirname = os.path.splitext(to_fn)[0] + "_imdctiles"
        if not os.path.isdir(to_tiles_dirname):
            os.mkdir(to_tiles_dirname)
        to_fn_tmps = []
        for fn in tiles_fns:
            to_fn_tmp = changext(fn, "_imdc.tif")
            to_fn_tmp = changefiledirname(to_fn_tmp, to_tiles_dirname)
            to_fn_tmps.append(to_fn_tmp)
            print("Image:", fn)
            print("Imdc :", to_fn_tmp)
            if os.path.isfile(to_fn_tmp):
                print("Imdc 100%")
                continue
            self.imdc(to_fn_tmp, geo_fns=[fn], data_deal=data_deal, is_jdt=is_jdt)
        print("Merge:", to_fn)
        gdal_merge_main(["gdal_merge_main",
                         "-of", "GTiff",
                         "-n", "0",
                         "-ot", "Byte",
                         "-co", "COMPRESS=PACKBITS",
                         "-o", to_fn,
                         *to_fn_tmps, ])
        tiffAddColorTable(to_fn, code_colors=self.color_table)

    def predict(self, x, *args, **kwargs):
        return self.clf.predict(x)

    def predictDF(self, df):
        x = df[self.x_keys]
        return self.predict(x)

    def randomSplitTrainTest(self, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None, ):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, train_size=train_size, random_state=random_state,
            shuffle=shuffle, stratify=stratify, )

    def testFieldSplitTrainTest(self, field_name="TEST", train_code=1, test_code=0):
        if self.select_list is not None:
            df = self.df[self.select_list]
        else:
            df = self.df
        train_select_list = df[field_name].values == train_code
        self.x_train, self.y_train = self.x[train_select_list], self.y[train_select_list]
        test_select_list = df[field_name].values == test_code
        self.x_test, self.y_test = self.x[test_select_list], self.y[test_select_list]

    def scoreCM(self, x, y, cm=None, category_names=None, map_dict=None):
        y1 = self.predict(x)
        if category_names is None:
            category_names = self.category_names[1:]
        cm = yToCM(y, y1, cm, category_names, map_dict)
        return cm

    def scoreTrainCM(self, cm=None, category_names=None, map_dict=None):
        self.train_cm = self.scoreCM(self.x_train.values, self.y_train, cm=cm, category_names=category_names,
                                     map_dict=map_dict)
        return self.train_cm

    def scoreTestCM(self, cm=None, category_names=None, map_dict=None):
        self.test_cm = self.scoreCM(self.x_test.values, self.y_test, cm=cm, category_names=category_names,
                                    map_dict=map_dict)
        return self.test_cm

    def saveModel(self, filename):
        joblib.dump(self.clf, filename)

    def loadModel(self, filename):
        self.clf = joblib.load(filename)


class MI_PytorchTraining(PytorchTraining):

    def __init__(self, model_dir=None, model_name="PytorchModel", epochs=10, device=None, n_test=100):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model = None
        self.models = []
        self._log = None
        super().__init__(epochs, device, n_test)
        self.cm_log = ConfusionMatrixLog()

        self.func_logit_category = funcPredict
        self.func_loss_deal = lambda loss: loss
        self.func_xy_deal = lambda x, y: (x.float(), y.long())
        self.func_y_deal = lambda y: y

        self.func_batch = None
        self.epoch_batch = None

    def toDict(self):
        to_dict_1 = super(MI_PytorchTraining, self).toDict()
        to_dict = {
            **to_dict_1,
            "cm_log": self.cm_log.toDict(),
            "func_logit_category": funcCodeString(self.func_logit_category),
            "func_loss_deal": funcCodeString(self.func_loss_deal),
            "func_xy_deal": funcCodeString(self.func_xy_deal),
            "func_y_deal": funcCodeString(self.func_y_deal),
        }
        return to_dict

    def initLog(self, log: TrainLog):
        self.cm_log.log = log
        self._log = log
        self._log.addField("ModelName", "string")
        self._log.addField("Epoch", "int")
        self._log.addField("Batch", "int")
        self._log.addField("Loss", "float")
        self._log.printOptions(print_float_decimal=3)

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):
        self.initTrain()

        for epoch in range(self.epochs):

            if self.epoch_batch is not None:
                self.epoch_batch()

            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x, y = self.func_xy_deal(x, y)

                if self.func_batch is not None:
                    self.func_batch()

                self.model.train()
                logts = self.model(x)
                self.loss = self.criterion(logts, y)
                self.loss = self.func_loss_deal(self.loss)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self.batchTAcc(batch_save, batchix, epoch)

            self.epochTAcc(epoch, epoch_save)
            if self.scheduler is not None:
                self.scheduler.step()

    def log(self, batch, epoch):
        self._log.updateField("Epoch", epoch + 1)
        self._log.updateField("Batch", batch + 1)
        self._log.updateField("Loss", self.loss.item())
        if batch == -1:
            model_name = "{0}_epoch{1}.pth".format(self.model_name, epoch)
        else:
            model_name = "{0}_epoch{1}_batch{2}.pth".format(self.model_name, epoch, batch)
        self._log.updateField("ModelName", model_name)

        if self.test_loader is not None:
            self.cm_log.updateLog("Test")

        self._log.saveLine()
        self._log.print(is_to_file=True)
        self._log.newLine()

        return model_name

    def tlogsave(self, is_save, batchix, epoch, is_print=False):
        if self.test_loader is not None:
            self.testAccuracy()
        modname = self.log(batchix, epoch)
        if is_save:
            mod_fn = self.saveModel(modname)
            if is_print:
                print("MODEL:", mod_fn)

    def batchTAcc(self, batch_save, batchix, epoch):
        if batchix % self.n_test == 0:
            self.tlogsave(batch_save, batchix, epoch, False)

    def epochTAcc(self, epoch, epoch_save):
        print("-" * 80)
        self.tlogsave(epoch_save, -1, epoch, True)
        print("*" * 80)

    def testAccuracy(self):
        self.cm_log.cms["Test"].clear()
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device).float()
                y = y.numpy()
                y = self.func_y_deal(y)
                y1 = self.func_logit_category(self.model, x)
                y1 = y1.cpu().numpy()
                self.cm_log.cms["Test"].addData(y, y1)
        self.model.train()
        self.cm_log.updateLog("Test")
        return self.cm_log.cms["Test"].OA()


class SRTModImPytorch(SRTModelImageInit):

    def __init__(
            self,
            model_dir=None,
            model_name="PytorchModel",
            epochs=100,
            device="cuda",
            n_test=100,
            batch_size=32,
            n_class=2,
            class_names=None,
            win_size=(),
    ):
        super().__init__()

        self.model_dirname = model_dir
        self.model_name = model_name
        self.epochs = epochs
        self.device = device
        self.n_test = n_test
        self.batch_size = batch_size
        self.n_class = n_class
        self.class_names = class_names
        self.win_size = win_size

        self.pt = MI_PytorchTraining()
        self.log = TrainLog()

        self.train_ds = None
        self.test_ds = None

        self.model = None

        self.func_predict = funcPredict
        self.func_y_deal = lambda y: y + 1

    def print(self):
        print(
            "model_dirname", self.model_dirname,
            "\nmodel_name", self.model_name,
            "\nepochs", self.epochs,
            "\ndevice", self.device,
            "\nn_test", self.n_test,
            "\nbatch_size", self.batch_size,
            "\nn_class", self.n_class,
            "\nclass_names", self.class_names,
            "\nwin_size", self.win_size,
        )
        if self.train_ds is not None:
            print("length of train_ds:", len(self.train_ds))
        if self.test_ds is not None:
            print("length of test_ds:", len(self.test_ds))
        self.test_ds = None

    def toDict(self):
        to_dict = {
            "model_name": self.model_name,
            "model_dirname": self.model_dirname,
            "category_names": self.category_names,
            "category_colors": self.category_colors,
            "ext_funcs": str(self.ext_funcs),
            "color_table": self.color_table,
            "train_cm": self.train_cm.toDict(),
            "test_cm": self.test_cm.toDict(),
            "epochs": self.epochs,
            "device": self.device,
            "n_test": self.n_test,
            "batch_size": self.batch_size,
            "n_class": self.n_class,
            "class_names": self.class_names,
            "win_size": self.win_size,
            "pt": self.pt.toDict(),
            "log": self.log.toDict(),
            "train_ds": pytorchModelCodeString(self.train_ds),
            "test_ds": pytorchModelCodeString(self.test_ds),
            "model": pytorchModelCodeString(self.model),
            "mod_fn": self.mod_fn,
            "func_predict": funcCodeString(self.func_predict),
            "func_y_deal": funcCodeString(self.func_y_deal),
        }
        return to_dict

    def initTrainLog(self):
        self.log.log_filename = os.path.join(self.model_dirname, "{0}_log.txt".format(self.model_name))
        self.log.save_csv_file = os.path.join(self.model_dirname, "{0}_log.csv".format(self.model_name))

    def initPytorchTraining(self):
        self.pt.__init__(
            model_dir=self.model_dirname,
            model_name=self.model_name,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test,
        )
        self.pt.initLog(self.log)
        self.pt.func_logit_category = self.func_predict

    def initDataLoader(self, train_ds=None, test_ds=None):
        if train_ds is None:
            train_ds = self.train_ds
        if test_ds is None:
            test_ds = self.test_ds
        self.pt.trainLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.pt.testLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def initModel(self, model=None):
        if model is None:
            model = self.model
        else:
            self.model = model
        model.to(self.device)
        self.pt.addModel(model)

    def initCriterion(self, criterion):
        self.pt.addCriterion(criterion)

    def initOptimizer(self, optimizer_class, *args, **kwargs):
        optimizer = optimizer_class(self.pt.model.parameters(), *args, **kwargs)
        self.pt.addOptimizer(optimizer=optimizer)

    def timeDirName(self):
        self.model_dirname = timeDirName(self.model_dirname, is_mk=True)

    def toCSVFN(self):
        return os.path.join(self.model_dirname, "train_data.csv")

    def copyFile(self, fn):
        to_fn = FN(fn).changedirname(self.model_dirname)
        copyfile(fn, to_fn)

    def train(self, *args, **kwargs):
        self.test_cm = ConfusionMatrix(n_class=self.n_class, class_names=self.class_names)
        self.pt.cm_log.addCM("Test", self.test_cm)
        self.pt.cm_log.initLog("Test")
        self.pt.cm_log.log.printOptions(print_type="keyword", print_field_names=["Epoch", "Batch", "Loss", "OATest"])
        smip_json_fn = os.path.join(self.model_dirname, "smip.json")
        print("smip_json_fn:", smip_json_fn)
        saveJson(self.toDict(), smip_json_fn, )
        self.pt.train()

    def imdc(self, to_geo_fn=None, geo_fns=None, grc: GDALRasterChannel = None, is_jdt=True, data_deal=None,
             is_print=True, data=None, gr=None, description="Category"):

        to_geo_fn = self.getToGeoFn(to_geo_fn)

        if is_print:
            print("to_geo_fn:", to_geo_fn)
        grc = self.initImdc1(geo_fns, grc)
        data = grc.fieldNamesToData(*list(grc.data.keys()))
        gr = list(grc.GRS.values())[0]

        imdc = self.imdcData(data, data_deal, is_jdt)
        self.saveImdc(to_geo_fn, gr, imdc, description)

    def saveImdc(self, to_geo_fn, gr, imdc, description):
        saveGTIFFImdc(gr, imdc, to_geo_fn, color_table=self.color_table, description=description)

    def imdcData(self, data, data_deal=None, is_jdt=True):
        if data_deal is not None:
            data = data_deal(data)
        self.model.eval()

        def func_predict(x):
            with torch.no_grad():
                x = torch.from_numpy(x).float().to(self.device)
                y = self.func_predict(self.model, x)
            y = y.cpu().numpy()
            return y

        imdc = dataPredictPatch(data, self.win_size, predict_func=func_predict, is_jdt=is_jdt)
        return imdc

    def imdcGDALFile(self, fn, to_fn, data_deal=None, is_jdt=True, description="CATEGORY"):
        gr = GDALRaster(fn)
        data = gr.readAsArray()
        imdc = self.imdcData(data, data_deal=data_deal, is_jdt=is_jdt)
        self.saveImdc(to_fn, gr, imdc, description=description)

    def imdcTiles(self, to_fn=None, tiles_dirname=None, tiles_fns=None, data_deal=None, is_jdt=True,
                  description="CATEGORY"):
        to_fn = self.getToGeoFn(to_fn)
        if tiles_fns is None:
            tiles_fns = []
        tiles_fns = list(tiles_fns)
        if tiles_dirname is not None:
            tiles_fns.extend(filterFileExt(tiles_dirname, ext=".tif"))
        to_tiles_dirname = os.path.splitext(to_fn)[0] + "_imdctiles"
        if not os.path.isdir(to_tiles_dirname):
            os.mkdir(to_tiles_dirname)
        to_fn_tmps = []
        for fn in tiles_fns:
            to_fn_tmp = changext(fn, "_imdc.tif")
            to_fn_tmp = changefiledirname(to_fn_tmp, to_tiles_dirname)
            to_fn_tmps.append(to_fn_tmp)
            print("Image:", fn)
            print("Imdc :", to_fn_tmp)
            if os.path.isfile(to_fn_tmp):
                print("Imdc 100%")
                continue
            self.imdcGDALFile(fn, to_fn_tmp, data_deal=data_deal, is_jdt=is_jdt, description=description)
        print("Merge:", to_fn)
        gdal_merge_main(["gdal_merge_main",
                         "-of", "GTiff",
                         "-n", "0",
                         "-ot", "Byte",
                         "-co", "COMPRESS=PACKBITS",
                         "-o", to_fn,
                         *to_fn_tmps, ])
        tiffAddColorTable(to_fn, code_colors=self.color_table)

    def predict(self, x, *args, **kwargs):
        return self.func_predict(self.model, x)

    def loadPTH(self, mod_fn):
        if mod_fn is None:
            mod_fn = sys.argv[3]
        data = torch.load(mod_fn)
        self.model.load_state_dict(data)
        self.mod_fn = mod_fn


class GDALImdcTiles:

    def __init__(self, *tiles_fns):
        self.fns = datasCaiFen(tiles_fns)
        self.grs = {}
        self.initTiles(*tiles_fns)

    def initTiles(self, *tiles_fns):
        self.fns = datasCaiFen(tiles_fns)
        for fn in self.fns:
            fn = os.path.abspath(fn)
            gr = GDALRaster(fn)
            self.grs[fn] = gr

    def imdc(self, model, to_imdc_fn, fit_keys=None, data_deal=None, is_jdt=True, color_table=None):
        to_imdc_dirname = changext(to_imdc_fn, "_tiles")

        if not os.path.isdir(to_imdc_dirname):
            os.mkdir(to_imdc_dirname)
        to_fn_tmps = []

        for fn in self.grs:
            gr = self.grs[fn]
            data = np.zeros((len(fit_keys), gr.n_rows, gr.n_columns))
            for i, k in enumerate(fit_keys):
                data[i] = gr.readGDALBand(k)
            gr.d = data
            to_imdc_fn_tmp = os.path.join(to_imdc_dirname, changext(os.path.split(fn)[1], "_imdc.tif"))
            to_fn_tmps.append(to_imdc_fn_tmp)
            imdc = dataModelPredict(gr.d, data_deal=data_deal, is_jdt=is_jdt, model=model)
            gr.save(imdc.astype("int8"), to_imdc_fn_tmp, fmt="GTiff", dtype=gdal.GDT_Byte,
                    options=["COMPRESS=PACKBITS"])
            if color_table is not None:
                tiffAddColorTable(to_imdc_fn_tmp, 1, color_table)
            del gr.d
            del data
            gr.d = None

        gdal_merge_main(["gdal_merge_main",
                         "-of", "GTiff",
                         "-n", "0",
                         "-ot", "Byte",
                         "-co", "COMPRESS=PACKBITS",
                         "-o", to_imdc_fn,
                         *to_fn_tmps, ])
        if color_table is not None:
            tiffAddColorTable(to_imdc_fn, 1, code_colors=color_table)


def imdc1(model, data, to_geo_fn, gr, data_deal=None, is_jdt=True, color_table=None):
    imdc = dataModelPredict(data, data_deal=data_deal, is_jdt=is_jdt, model=model)
    gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
    if color_table is not None:
        tiffAddColorTable(to_geo_fn, 1, color_table)


def imdc2(func_predict, data, win_size, to_geo_fn, gr, data_deal=None, is_jdt=True, color_table=None):
    if data is not None:
        data = data_deal(data)
    imdc = dataPredictPatch(data, win_size, func_predict, is_jdt=is_jdt)
    gr.save(imdc.astype("int8"), to_geo_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
    if color_table is not None:
        tiffAddColorTable(to_geo_fn, 1, color_table)


class GDALImdc:

    def __init__(self, *raster_fns):
        self.raster_fns = datasCaiFen(raster_fns)
        self.color_table = None
        self.sfm = SRTFeaturesMemory()
        if len(self.raster_fns) >= 1:
            self.sfm = SRTFeaturesMemory(names=GDALRaster(raster_fns[0]).names)

    def imdc1(self, model, to_imdc_fn, fit_names=None, data_deal=None, is_jdt=True, color_table=None):
        color_table, fit_names = self._initImdc(color_table, fit_names)

        if len(self.raster_fns) == 1:
            raster_fn = self.raster_fns[0]
            self._imdc1(model, raster_fn, to_imdc_fn, fit_names, data_deal, is_jdt, color_table)
        else:
            to_imdc_dirname = changext(to_imdc_fn, "_tiles")
            if not os.path.isdir(to_imdc_dirname):
                os.mkdir(to_imdc_dirname)

            to_fn_tmps = []
            for fn in self.raster_fns:
                to_imdc_fn_tmp = os.path.join(to_imdc_dirname, changext(os.path.split(fn)[1], "_imdc.tif"))
                to_fn_tmps.append(to_imdc_fn_tmp)
                self._imdc1(model, fn, to_imdc_fn_tmp, fit_names, data_deal, is_jdt, color_table)

            gdal_merge_main(["gdal_merge_main", "-of", "GTiff", "-n", "0", "-ot", "Byte", "-co", "COMPRESS=PACKBITS",
                             "-o", to_imdc_fn, *to_fn_tmps, ])

            if color_table is not None:
                tiffAddColorTable(to_imdc_fn, 1, code_colors=color_table)

    def _initImdc(self, color_table, fit_names):
        if fit_names is None:
            fit_names = self.sfm.names
        if color_table is None:
            color_table = self.color_table
        if len(self.raster_fns) == 0:
            raise Exception("Can not find raster")
        return color_table, fit_names

    def _imdc1(self, model, raster_fn, to_geo_fn, fit_names, data_deal, is_jdt, color_table):
        gr = GDALRaster(raster_fn)
        data = np.zeros((len(fit_names), gr.n_rows, gr.n_columns))
        jdt = Jdt(len(fit_names), "Read Raster").start(is_jdt)
        for i, name in enumerate(fit_names):
            # data[i] = self.sfm.callbacks(name).fit(gr.readGDALBand(name))
            data_i =  gr.readGDALBand(name)
            data_i[np.isnan(data_i)] = 0
            data[i] = data_i
            jdt.add(is_jdt)
        jdt.end(is_jdt)
        imdc1(model, data, to_geo_fn, gr, data_deal=data_deal, is_jdt=is_jdt, color_table=color_table)
        data = None

    def _imdc2(self, func_predict, raster_fn, win_size, to_geo_fn, fit_names, data_deal, is_jdt, color_table):
        gr = GDALRaster(raster_fn)
        data = np.zeros((len(fit_names), gr.n_rows, gr.n_columns))
        for i, name in enumerate(fit_names):
            data[i] = self.sfm.callbacks(name).fit(gr.readGDALBand(name))
        imdc2(func_predict, data, win_size=win_size, to_geo_fn=to_geo_fn, gr=gr, data_deal=data_deal, is_jdt=is_jdt,
              color_table=color_table)
        data = None

    def imdc2(self, func_predict, win_size, to_imdc_fn, fit_names, data_deal=None, is_jdt=True, color_table=None):
        color_table, fit_names = self._initImdc(color_table, fit_names)
        if len(self.raster_fns) == 1:
            raster_fn = self.raster_fns[0]
            self._imdc2(func_predict, raster_fn, win_size, to_imdc_fn, fit_names, data_deal, is_jdt, color_table)
        else:
            to_imdc_dirname = changext(to_imdc_fn, "_tiles")
            if not os.path.isdir(to_imdc_dirname):
                os.mkdir(to_imdc_dirname)

            to_fn_tmps = []
            for fn in self.raster_fns:
                to_imdc_fn_tmp = os.path.join(to_imdc_dirname, changext(os.path.split(fn)[1], "_imdc.tif"))
                to_fn_tmps.append(to_imdc_fn_tmp)
                self._imdc2(func_predict, fn, win_size, to_imdc_fn, fit_names, data_deal, is_jdt, color_table)

            gdal_merge_main(["gdal_merge_main", "-of", "GTiff", "-n", "0", "-ot", "Byte", "-co", "COMPRESS=PACKBITS",
                             "-o", to_imdc_fn, *to_fn_tmps, ])

            if color_table is not None:
                tiffAddColorTable(to_imdc_fn, 1, code_colors=color_table)


def main():
    return


if __name__ == "__main__":
    main()
