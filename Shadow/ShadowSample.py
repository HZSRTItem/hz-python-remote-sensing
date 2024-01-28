# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowSample.py
@Time    : 2023/8/31 14:54
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowSample
样本基础设定n个样本，之后同时逐步加入一定数量的样本
特征组合类型加几种
分类器加几种
-----------------------------------------------------------------------------"""
import os

import joblib
import numpy as np
import pandas as pd
from osgeo import osr

from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.GDALUtils import samplingSingle
from SRTCodes.NumpyUtils import saveCM, npShuffle2, changePandasIndex
from SRTCodes.SRTSample import CSVSamples, SRTSampleSelect
from SRTCodes.Utils import RumTime, printList, DirFileName, savecsv, Jdt
from Shadow.ShadowFeatureCallback import bjFeatureCallback, qdFeatureCallback, cdFeatureCallback
from Shadow.ShadowTraining import FrontShadowCategoryTraining


def sampleDataShuffle(x, y):
    d = np.concatenate([x, np.array([y]).T], axis=1)
    d = npShuffle2(d)
    return d[:, :-1], d[:, -1]


class ShadowSampleNumber(FrontShadowCategoryTraining):

    def __init__(self, model_dir, n_category, category_names):
        super(ShadowSampleNumber, self).__init__(model_dir, n_category=n_category,
                                                 category_names=category_names)

        self.sh_train_y = None
        self.sh_train_x = None
        self.nosh_train_y = None
        self.nosh_train_x = None

        self.sample_numbers = []
        self.init_number = 0
        self.feature_types = {}

        self.test_x = None
        self.test_y = None
        self.train_x = None
        self.train_y = None

        self.save_cm_file = None
        self.run_time = RumTime(len(self.mod_types) * len(self.sample_numbers) * len(self.feature_types))
        self.logAddField()

    def logAddField(self):
        self._log.addField("MOD_TYPE", "string")
        self._log.addField("SPL_TYPE", "string")
        self._log.addField("FEAT_TYPE", "string")
        self._log.addField("SPL_NUMBER", "int")
        self._log.addField("TRAIN_CM", "int")
        self._log.addField("TEST_CM", "int")
        self._log.addField("NUMBER", "int")
        self._log.printOptions(
            print_type="keyword",
            print_float_decimal=3,
            print_sep="\n",
            print_field_names=["OATrain", "KappaTrain", "OATest", "KappaTest", "MOD_TYPE", "SPL_NUMBER", "FEAT_TYPE"]
        )

    def setSample(self, csv_spl: CSVSamples):
        self.csv_spl = csv_spl
        x, y = self.csv_spl.get()
        select = x[self.test_field_name].values
        # test sample
        self.test_x = x[select == 0]
        self.test_y = y[select == 0]
        y1 = self.test_y[self.test_y >= 5]
        self.test_y[self.test_y >= 5] = y1 - 4
        # train sample
        self.train_x = x[select == 1]
        self.train_y = y[select == 1]

    def sampleNumbers(self, init_number, add_number, add_split):
        self.init_number = init_number
        for i in range(add_split):
            self.sample_numbers.append(i * add_number)

    def sampleRandom(self):
        """ get random sample not features split"""
        n_train_spl = len(self.train_y)
        df = self.train_x
        df["__Y__"] = self.train_y
        d = npShuffle2(df.values)
        self.train_x = pd.DataFrame(d, columns=list(df.keys()))
        self.train_y = self.train_x["__Y__"].values

        self.nosh_train_x = changePandasIndex(self.train_x[self.train_y < 5].copy())
        self.nosh_train_y = self.train_y[self.train_y < 5].copy()
        self.sh_train_x = changePandasIndex(self.train_x[self.train_y >= 5].copy())
        self.sh_train_y = self.train_y[self.train_y >= 5].copy()
        self.sh_train_y = self.sh_train_y - 4

        x = self.nosh_train_x.loc[:self.init_number - 1].values
        y = self.nosh_train_y[:self.init_number]
        column_names = list(df.keys())

        for i in range(len(self.sample_numbers)):
            n = self.sample_numbers[i]
            if n < 0:
                raise Exception("Sample Number have to more than 0 {0}:{1}".format(i, n))
            if n != 0:
                n += self.init_number
                sh_x = self.sh_train_x.loc[self.init_number:n - 1].values
                sh_y = self.sh_train_y[self.init_number:n]
                no_sh_x = self.nosh_train_x.loc[self.init_number:n - 1].values
                no_sh_y = self.nosh_train_y[self.init_number:n]
                sh_x = np.concatenate([x, sh_x])
                sh_y = np.concatenate([y, sh_y])
                sh_x, sh_y = sampleDataShuffle(sh_x, sh_y)
                no_sh_x = np.concatenate([x, no_sh_x])
                no_sh_y = np.concatenate([y, no_sh_y])
                no_sh_x, no_sh_y = sampleDataShuffle(no_sh_x, no_sh_y)

                self.sample_numbers[i] = {
                    "no_sh_x": pd.DataFrame(no_sh_x, columns=column_names), "no_sh_y": no_sh_y,
                    "sh_x": pd.DataFrame(sh_x, columns=column_names), "sh_y": sh_y,
                    "n": n}
            else:
                self.sample_numbers[i] = {
                    "no_sh_x": pd.DataFrame(x, columns=column_names), "no_sh_y": y,
                    "sh_x": pd.DataFrame(x, columns=column_names), "sh_y": y,
                    "n": self.init_number}

    def getSample(self, feat_types, spl, spl_type="sh"):
        x_test = self.test_x[feat_types].values
        y_test = self.test_y
        if spl_type == "sh":
            x_train = spl["sh_x"][feat_types].values
            y_train = spl["sh_y"]
        else:
            x_train = spl["no_sh_x"][feat_types].values
            y_train = spl["no_sh_y"]
        return x_train, y_train, x_test, y_test

    def saveModel(self, model_name, *args, **kwargs):
        if model_name is not None:
            joblib.dump(self.model, model_name)

    def train(self):
        self.timeModelDir()
        self.saveDataToModelDirectory()
        self.save_cm_file = os.path.join(self.model_dir, "cm.txt")

        self._log.newLine()
        self._log.printFirstLine(is_to_file=True)
        self._log.saveHeader()
        self._initFeatures()

        self.run_time = RumTime(len(self.mod_types) * len(self.sample_numbers) * len(self.feat_types))
        self.run_time.strat()
        n_run = 1

        print(self.model_dir)
        for model_type in self.mod_types:
            mod_train = self.mod_types[model_type]
            for feat_name in self.feat_types:
                feat_types = self.feat_types[feat_name]
                for spl in self.sample_numbers:
                    for spl_type in ["sh", "no_sh"]:
                        # model name
                        mod_names = ["NSPL_{0}".format(spl["n"]), model_type, feat_name]
                        mod_name = "-".join(mod_names)
                        mod_fn = os.path.join(self.model_dir, mod_name + "_mod.model")
                        mod_args_fn = os.path.join(self.model_dir, mod_name + "_args.json")
                        print("{0}. {1}".format(n_run, " ".join(mod_names)))
                        # train running ---
                        x_train, y_train, x_test, y_test = self.getSample(feat_types, spl, spl_type)
                        mod, mod_args = mod_train(x_train, y_train)
                        self.model = mod
                        self.mod_args = mod_args
                        self.saveModel(mod_fn)
                        self.mod_args["model_name"] = mod_name
                        self.mod_args["model_filename"] = mod_fn
                        self.mod_args["features"] = feat_types.copy()
                        self.mod_args["number"] = n_run
                        self.saveModArgs(mod_args_fn)
                        # update confusion matrix
                        y_train_2 = self.model.predict(x_train)
                        self.train_cm.addData(y_train, y_train_2)
                        self.updateLogTrainCM()
                        y_test_2 = self.model.predict(x_test)
                        self.test_cm.addData(y_test, y_test_2)
                        self.updateLogTestCM()
                        # save confusion matrix
                        train_cm_arr = self.train_cm.calCM()
                        n = saveCM(train_cm_arr, self.save_cm_file, cate_names=self.category_names,
                                   infos=["TRAIN"] + mod_names)
                        self._log["TRAIN_CM"] = n
                        test_cm_arr = self.test_cm.calCM()
                        n = saveCM(test_cm_arr, self.save_cm_file, cate_names=self.category_names,
                                   infos=["TEST"] + mod_names)
                        self._log["TEST_CM"] = n
                        # update log
                        self._log["MOD_TYPE"] = model_type.upper()
                        self._log["SPL_TYPE"] = spl_type.upper()
                        self._log["FEAT_TYPE"] = feat_name.upper()
                        self._log["SPL_NUMBER"] = spl["n"]
                        self._log["NUMBER"] = n_run
                        self._log["ModelName"] = mod_name
                        self._log.saveLine()
                        self._log.print(is_to_file=True)
                        self._log.newLine()
                        # update run time
                        self.run_time.add()
                        self.run_time.printInfo()
                        n_run += 1
                        print()
                        # clear confusion matrix
                        self.train_cm.clear()
                        self.test_cm.clear()

    def print(self):
        super(ShadowSampleNumber, self).print()
        printList("Train Log Fields:", [field for field in self._log])
        self.csv_spl.print()
        print()
        printList("- Model Types: ", [k for k in self.mod_types])
        print()
        for k in self.feat_types:
            printList("* Feature Types [{0}]:".format(k), self.feat_types[k])
        print()
        for k in self.tag_types:
            printList("* Tag Types [{0}]:".format(k), self.tag_types[k])
        print()
        print("* Sample Number:")
        for k in range(len(self.sample_numbers)):
            print("  - {0}:{1}".format(k, self.sample_numbers[k]["n"]))


class ShadowCSVSample:

    def __init__(self, csv_fn=None):
        self.csv_fn = csv_fn
        self.csv_spl = CSVSamples()
        self.spl_select = SRTSampleSelect()

        self.x = None
        self.y = None

        self.feats = None
        self.categorys = None
        self.tags = None

        self.field_c_name = "CNAME"
        self.field_tag_name = "TAG"
        self.c_names = ["NOT_KNOW", "IS", "VEG", "SOIL", "WAT", "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH"]

        self.init()

    def init(self, csv_fn=None):
        if csv_fn is None:
            csv_fn = self.csv_fn
        self.csv_spl = CSVSamples(csv_fn)
        self.feats = self.csv_spl.getFeatureNames()
        self.categorys = self.csv_spl.getCategoryNames()
        self.tags = self.csv_spl.getTagNames()

    def fieldNameCategory(self, cname):
        self.field_c_name = cname

    def fieldNameTag(self, tag_name):
        self.field_tag_name = tag_name

    def addCategoryNames(self, *cnames):
        self.c_names = list(cnames)

    def featureScaleMinMax(self, feat_name, x_min, x_max, is_trans=None, is_01=None):
        self.csv_spl.featureScaleMinMax(feat_name=feat_name, x_min=x_min, x_max=x_max, is_trans=is_trans, is_01=is_01)

    def featureCallBack(self, feat_name, callback_func, is_trans=None):
        self.csv_spl.featureCallBack(feat_name=feat_name, callback_func=callback_func, is_trans=is_trans)

    def readData(self):
        self.csv_spl.fieldNameCategory(self.field_tag_name)  # CNAME
        self.csv_spl.fieldNameTag(self.field_tag_name)
        self.csv_spl.addCategoryNames(self.c_names)
        self.csv_spl.readData()

    def get(self, c_names=None, feat_names=None, tags=None):
        self.x, self.y = self.csv_spl.get(c_names=c_names, feat_names=feat_names, tags=tags)

    def initSelect(self, x=None, y=None, sampling_type="no_back"):
        """
        sampling_type: Is the sample a sample that has been returned or not been returned `back|no_back`
        """
        if x is None:
            x = self.x.copy()
        if y is None:
            y = self.y.copy()
        self.spl_select = SRTSampleSelect(x=x, y=y, sampling_type=sampling_type)

    def select(self, category_number_dict, map_dict=None, fields=None):
        df, y = self.spl_select.get(category_number_dict=category_number_dict, map_dict=map_dict)
        if fields is not None:
            df = df[fields]
        return df, y

    def __len__(self):
        self.csv_spl.__len__()

    def sampling(self, raster_fn, to_csv_fn, x_field="X", y_field="Y", coor_srs="EPSG:4326"):
        d, gr, x, y = self._initSampling(raster_fn, x_field, y_field)
        d = samplingSingle(x, y, coor_srs, gr, data_dict=d)
        savecsv(to_csv_fn, d)

    def _initSampling(self, raster_fn, x_field, y_field):
        d = self.csv_spl.getDFData()
        d = {k: d[k].tolist() for k in d}
        x = list(map(float, d[x_field]))
        y = list(map(float, d[y_field]))
        gr = GDALRaster(raster_fn)
        return d, gr, x, y

    def samplingCenter(self, raster_fn, to_fn, spl_size, channel_list=None,
                       x_field="X", y_field="Y", coor_srs="EPSG:4326", is_jdt=True):
        d, gr, x, y = self._initSampling(raster_fn, x_field, y_field)
        select_list = []
        d_list = []

        srs = osr.SpatialReference()
        srs.SetFromUserInput(coor_srs)
        gr.setDstSrs(srs)

        jdt = Jdt(len(x), "Sampling Center")
        if is_jdt:
            jdt.start()
        for i in range(len(x)):
            x0, y0 = x[i], y[i]
            d0 = gr.readAsArrayCenter(x0, y0, win_row_size=spl_size[0], win_column_size=spl_size[1],
                                      interleave="band", is_geo=True, is_trans=True)
            if channel_list is not None:
                d0 = d0[channel_list]
            if d0 is not None:
                select_list.append(1)
                d_list.append([d0])
            else:
                select_list.append(0)
            if is_jdt:
                jdt.add()
        if is_jdt:
            jdt.end()

        d["SPLC_SEL"] = select_list
        savecsv(to_fn + ".csv", d)
        d_list = np.concatenate(d_list)
        np.save(to_fn + ".csv", d_list)


class ShadowCSVSampleCollection:

    def __init__(self, is_init_all=False, is_call_back_all=False, is_read_data=False,
                 qd_csv_fn=None, bj_csv_fn=None, cd_csv_fn=None,
                 qd_raster_fn=None, bj_raster_fn=None, cd_raster_fn=None, ):
        self.qd = ShadowCSVSample()
        self.bj = ShadowCSVSample()
        self.cd = ShadowCSVSample()

        dfn = DirFileName(r"F:\ProjectSet\Shadow\Release")

        self.qd_csv_fn = dfn.fn(r"QingDaoSamples\sh_qd_sample_spl.csv") if qd_csv_fn is None else qd_csv_fn
        self.bj_csv_fn = dfn.fn(r"BeiJingSamples\sh_bj_sample_spl.csv") if bj_csv_fn is None else bj_csv_fn
        self.cd_csv_fn = dfn.fn(r"ChengDuSamples\sh_cd_sample_spl.csv") if cd_csv_fn is None else cd_csv_fn

        self.qd_raster_fn = dfn.fn(r"QingDaoImages\SH_QD_envi.dat") if qd_csv_fn is None else qd_raster_fn
        self.bj_raster_fn = dfn.fn(r"BeiJingImages\SH_BJ_envi.dat") if bj_csv_fn is None else bj_raster_fn
        self.cd_raster_fn = dfn.fn(r"ChengDuImages\SH_CD_envi.dat") if cd_csv_fn is None else cd_raster_fn

        if is_init_all:
            self.initQD()
            self.initBJ()
            self.initCD()
            if is_read_data:
                self.qd.readData()
                self.bj.readData()
                self.cd.readData()
            if is_call_back_all:
                self.qdFeatureCallback()
                self.bjFeatureCallback()
                self.cdFeatureCallback()

    def initQD(self, csv_fn=None):
        if csv_fn is None:
            csv_fn = self.qd_csv_fn
        self.qd = ShadowCSVSample(csv_fn)

    def initBJ(self, csv_fn=None):
        if csv_fn is None:
            csv_fn = self.bj_csv_fn
        self.bj = ShadowCSVSample(csv_fn)

    def initCD(self, csv_fn=None):
        if csv_fn is None:
            csv_fn = self.cd_csv_fn
        self.cd = ShadowCSVSample(csv_fn)

    def bjFeatureCallback(self):
        bjFeatureCallback(self.bj)

    def qdFeatureCallback(self):
        qdFeatureCallback(self.bj)

    def cdFeatureCallback(self):
        cdFeatureCallback(self.bj)


class ShadowCSVSampleCollectionGDAL(ShadowCSVSampleCollection):

    def __init__(self, is_init_all=False, is_call_back_all=False, qd_csv_fn=None, bj_csv_fn=None, cd_csv_fn=None):
        super().__init__(is_init_all, is_call_back_all, qd_csv_fn, bj_csv_fn, cd_csv_fn)
