# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowImdC.py
@Time    : 2023/7/7 9:51
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of ShadowImdC
-----------------------------------------------------------------------------"""
import os.path

import joblib
import numpy as np

from SRTCodes.ENVIRasterClassification import ENVIRasterClassification
from SRTCodes.SRTFeature import SRTFeatureCallBackCollection
from SRTCodes.Utils import Jdt, RumTime, readJson, SRTMultipleOpening


class ShadowImageClassification(ENVIRasterClassification):

    def __init__(self, dat_fn, model_dir, is_trans=True, is_scale_01=True):
        super(ShadowImageClassification, self).__init__(dat_fn=dat_fn)
        self.is_trans = is_trans
        self.is_scale_01 = is_scale_01
        self._feat_callback_coll = SRTFeatureCallBackCollection(is_trans=is_trans)
        self.model_dir = model_dir
        self.smo = SRTMultipleOpening(os.path.join(self.model_dir, "ShadowImageClassification.smo"))

    def featureCallBack(self, feat_name, callback_func, is_trans=None):
        is_trans = self._featureCallBackCheck(feat_name, is_trans)
        self._feat_callback_coll.addCallBack(feat_name, callback_func=callback_func, is_trans=is_trans)

    def _featureCallBackCheck(self, feat_name, is_trans):
        if is_trans is None:
            is_trans = self.is_trans
        if feat_name not in self.names:
            raise Exception("Feature Name \"{0}\" not in names.".format(feat_name))
        return is_trans

    def featureScaleMinMax(self, feat_name, x_min, x_max, is_trans=None, is_01=None):
        is_trans = self._featureCallBackCheck(feat_name, is_trans)
        if is_01 is None:
            is_01 = self.is_scale_01
        self._feat_callback_coll.addScaleMinMax(feat_name, x_min, x_max, is_trans=is_trans, is_to_01=is_01)

    def predict(self, x, *args, **kwargs) -> np.ndarray:
        return self.model.predict(x)

    def preDeal(self, row, column_start=0, column_end=-1):
        return np.array(self.n_columns, dtype="bool")

    def run(self, *args, **kwargs):
        self._readData()

        json_fns = []
        for f in os.listdir(self.model_dir):
            json_fn = os.path.join(self.model_dir, f)
            if os.path.isfile(json_fn):
                if os.path.splitext(json_fn)[1] == ".json":
                    json_fns.append(json_fn)

        run_time = RumTime(len(json_fns))
        run_time.strat()

        for i, json_fn in enumerate(json_fns):
            mod_args = readJson(json_fn)
            mod_name = mod_args["model_name"]
            mod_fn = mod_args["model_filename"]
            features = mod_args["features"]

            print("{0}. {1}".format(i + 1, mod_name))
            to_f = self.classify(mod_fn, features, mod_name)
            print("--> {0}".format(to_f))

            run_time.add()
            run_time.printInfo()

        run_time.end()

    def classify(self, mod_fn, features, mod_name):
        print("MODEL NAME:", mod_fn)
        to_f = os.path.join(self.model_dir, mod_name + "_imdc.dat")
        if os.path.isfile(to_f):
            print("Shadow Image RasterClassification: 100%")
            return to_f
        if self.d is None:
            self._readData()
        self._initImdc()
        self.readModel(mod_fn)
        d = self.getFeaturesData(features)
        is_r = self.smo.is_run(to_f)
        if is_r == 1:
            print("Shadow Image RasterClassification: Running")
            return to_f
        elif is_r == -1:
            print("Shadow Image RasterClassification: 100% -- END")
            return to_f

        self.smo.add(to_f)
        jdt = Jdt(total=self.n_rows, desc="Shadow Image RasterClassification")
        jdt.start()
        for i in range(0, self.n_rows):
            col_imdc = d[:, i, :].T
            y = self.predict(col_imdc)
            self.imdc[i, :] = y
            jdt.add()
        jdt.end()
        self.saveImdc(to_f)
        self.smo.end(to_f)

        return to_f

    def print(self):
        print(self.names)

    def readModel(self, mod_fn=None):
        self.model = joblib.load(mod_fn)

    def _readData(self):
        self.d = self.readAsArray(interleave="b,r,c")
        for i, name in enumerate(self.names):
            if name in self._feat_callback_coll:
                self.d[i] = self._feat_callback_coll.fits(name, self.d[i])

    def getFeaturesData(self, features):
        idx_d = []
        for i, name in enumerate(features):
            if name == "TEST":
                continue
            idx_d.append(self.names.index(name))
        return self.d[idx_d, :]
