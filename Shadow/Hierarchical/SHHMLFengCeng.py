# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHMLFengCeng.py
@Time    : 2024/3/8 21:06
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHMLFengCeng
-----------------------------------------------------------------------------"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.GDALRasterIO import GDALRasterChannel, GDALRaster, tiffAddColorTable
from SRTCodes.GDALUtils import GDALRastersSampling
from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.Utils import timeDirName, DirFileName, SRTWriteText, saveJson, readJson, Jdt, filterFileExt, \
    SRTDFColumnCal, getfilenamewithoutext
from Shadow.Hierarchical import SHHConfig
from Shadow.Hierarchical.SHHConfig import SHH_COLOR8, categoryMap


def main():
    pass


if __name__ == "__main__":
    main()


def feat_norm(d1, d2):
    if isinstance(d1, np.ndarray):
        d1[np.isnan(d1)] = 1
        d1[np.isnan(d2)] = -1.0001
    return (d1 - d2) / (d1 + d2)


def ext_feat(data, *feat_names):
    for feat_name in feat_names:
        if feat_name == "ndvi":
            data[feat_name] = feat_norm(data["B8"], data["B4"])
        elif feat_name == "ndwi":
            data[feat_name] = feat_norm(data["B3"], data["B8"])
        elif feat_name == "mndwi":
            data[feat_name] = feat_norm(data["B3"], data["B12"])


def df_get_data(df, x_keys, test_n, fengcheng=-1, cate_name=None):
    if fengcheng != -1:
        if "TEST" in df:
            df_train = df[df["TEST"] == test_n]
        else:
            df_train = df
        df_train = df_train[df_train["FEN_CENG"] == fengcheng]
    else:
        if "TEST" in df:
            df_train = df[df["TEST"] == test_n]
        else:
            df_train = df
    x = df_train[x_keys].values
    if fengcheng != -1:
        if cate_name is None:
            cate_name = "FC"
        y = df_train[cate_name].values
    else:
        if cate_name is None:
            cate_name = "__CODE__"
        y = df_train[cate_name].values
    return x, y


def read_geo_raster(geo_fn, imdc_keys, glcm_fn=None):
    grc = GDALRasterChannel()
    grc.addGDALDatas(geo_fn)
    if glcm_fn is not None:
        grc.addGDALDatas(glcm_fn)
    ext_feat(grc, *imdc_keys)
    data = grc.fieldNamesToData(*imdc_keys)
    return data


def feature_importance(mod: RandomForestClassifier, show_keys):
    fi = mod.feature_importances_
    show_data = list(zip(show_keys, fi))
    print(show_data)
    show_data.sort(key=lambda _elem: _elem[1], reverse=False)
    print(show_data)
    show_data = dict(show_data)
    plt.barh(list(show_data.keys()), list(show_data.values()))
    plt.show()


def trainMLFC():
    # 分层方法测试，使用机器学习的方法
    # python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHTemp import method_name7; method_name7()"
    def get_data(x_keys, test_n, fengcheng=-1):
        return df_get_data(df, x_keys, test_n, fengcheng)

    """
    'SRT', 'OSRT', 'X', 'Y', 'CNAME', 'CATEGORY', 'TAG', 'TEST', 'CODE',
    'CITY', 'AS_VV', 'AS_VH', 'AS_angle', 'DE_VV', 'DE_VH', 'DE_angle',
    'B2', 'B3', 'B4', 'B8', 'B11', 'B12', '__CODE__', 'FEN_CENG', 'FC',
    'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent',
    'OPT_asm', 'OPT_cor', 'ndvi', 'ndwi', 'mndwi'
    """
    is_fenceng = "fc"
    init_dirname = timeDirName(r"F:\ProjectSet\Shadow\Hierarchical\MLMods", is_mk=False)
    init_dirname += is_fenceng
    if not os.path.isdir(init_dirname):
        os.mkdir(init_dirname)
    init_dfn = DirFileName(init_dirname)
    log_wt = SRTWriteText(init_dfn.fn("log.txt"))
    code_fn = os.path.split(__file__)[1]
    log_wt.write("CODE_FN", __file__)
    log_wt.write("CODE_TO_FN", code_fn)
    with open(init_dfn.fn(code_fn), "w", encoding="utf-8") as fw:
        with open(__file__, "r", encoding="utf-8") as fr:
            fw.write(fr.read())
    df_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\5\FenCengSamples_glcm.xlsx"
    df = pd.read_excel(df_fn, sheet_name="GLCM")
    log_wt.write("DF_FN:", df_fn)
    ext_feat(df, "ndvi", "ndwi", "mndwi")
    print(df.keys())
    log_wt.write("IS_FENCENG:", is_fenceng)
    nofc_x_keys = [
        'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
        'ndvi', 'ndwi', 'mndwi',
        'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH',
        'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm'
    ]
    log_wt.write("NOFC_X_KEYS:", nofc_x_keys)
    df.to_csv(init_dfn.fn("train_data.csv"))

    class fen_ceng:

        def __init__(self):
            self.veg_high_low_clf = RandomForestClassifier(150)
            self.vhl_keys = [
                'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
                'ndvi', 'ndwi', 'mndwi',
            ]
            log_wt.write("VEG_HIGH_LOW_CLF:", self.veg_high_low_clf)
            log_wt.write("VHL_KEYS:", self.vhl_keys)
            self.is_soil_clf = RandomForestClassifier(150)
            self.is_keys = [
                'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
                'AS_VV', 'AS_VH',
                'DE_VV', 'DE_VH',
                'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm',
                'ndvi', 'ndwi', 'mndwi',
            ]
            log_wt.write("IS_SOIL_CLF:", self.is_soil_clf)
            log_wt.write("IS_KEYS:", self.is_keys)
            self.wat_sh_clf = RandomForestClassifier(150)
            self.ws_keys = [
                'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
                'ndvi', 'ndwi', 'mndwi',
                'AS_VV', 'AS_VH',
                'DE_VV', 'DE_VH',
                'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm',
            ]
            log_wt.write("WAT_SH_CLF:", self.wat_sh_clf)
            log_wt.write("WS_KEYS:", self.ws_keys)
            self.imdc_keys = []
            self.vhl_index_list = []
            self.is_index_list = []
            self.ws_index_list = []

        def get_keys(self):
            to_keys = []

            def _get_keys(_list):
                for k in _list:
                    if k not in to_keys:
                        to_keys.append(k)

            _get_keys(self.vhl_keys)
            _get_keys(self.is_keys)
            _get_keys(self.ws_keys)
            return to_keys

        def init_imdc_keys(self, imdc_keys: list):
            self.imdc_keys = imdc_keys
            self.vhl_index_list = [self.imdc_keys.index(k) for k in self.vhl_keys]
            self.is_index_list = [self.imdc_keys.index(k) for k in self.is_keys]
            self.ws_index_list = [self.imdc_keys.index(k) for k in self.ws_keys]

        def fit(self):
            def _train(n, _clf, fit_keys, name):
                _x_train, _y_train = get_data(fit_keys, 1, n)
                _x_test, _y_test = get_data(fit_keys, 0, n)
                _clf.fit(_x_train, _y_train)
                print(name, "train acc:", _clf.score(_x_train, _y_train))
                log_wt.write(name, "train acc:", _clf.score(_x_train, _y_train))
                print(name, "test acc:", _clf.score(_x_test, _y_test))
                log_wt.write(name, "test acc:", _clf.score(_x_test, _y_test))

            _train(1, self.veg_high_low_clf, self.vhl_keys, "veg_high_low")
            _train(2, self.is_soil_clf, self.is_keys, "is_soil")
            _train(3, self.wat_sh_clf, self.ws_keys, "wat_sh")

        def predict(self, x):
            y1_d = self.veg_high_low_clf.predict(x[:, self.vhl_index_list])
            y2_d = self.is_soil_clf.predict(x[:, self.is_index_list])
            y3_d = self.wat_sh_clf.predict(x[:, self.ws_index_list])
            y = []
            for i in range(len(x)):
                y1, y2, y3 = y1_d[i], y2_d[i], y3_d[i]
                if y1 == 1:
                    y.append(2)
                elif y1 == 2:
                    if y2 == 1:
                        y.append(1)
                    elif y2 == 2:
                        y.append(3)
                    else:
                        y.append(0)
                elif y1 == 3:
                    if y3 == 5:
                        y.append(4)
                    else:
                        y.append(y3 + 4)
                else:
                    y.append(0)
            return np.array(y)

        def save(self, to_fn=None):
            if to_fn is None:
                to_fn = init_dfn.fn("fc")
            json_fn = to_fn + ".mod.json"
            vhl_fn = to_fn + "_vhl.mod"
            joblib.dump(self.veg_high_low_clf, vhl_fn)
            is_fn = to_fn + "_is.mod"
            joblib.dump(self.is_soil_clf, is_fn)
            ws_fn = to_fn + "_ws.mod"
            joblib.dump(self.wat_sh_clf, ws_fn)
            to_dict = {"VHL_FN": vhl_fn, "IS_FN": is_fn, "WS_FN": ws_fn}
            saveJson(to_dict, json_fn)
            return {to_fn: to_dict}

        def load(self, json_fn):
            json_dict = readJson(json_fn)
            self.veg_high_low_clf = joblib.load(json_dict["VHL_FN"])
            self.is_soil_clf = joblib.load(json_dict["IS_FN"])
            self.wat_sh_clf = joblib.load(json_dict["WS_FN"])

    def fenceng():
        _clf = fen_ceng()
        _clf.fit()
        print()
        return _clf

    def not_fenceng():
        print(nofc_x_keys)
        x_train, y_train = get_data(nofc_x_keys, 1)
        x_test, y_test = get_data(nofc_x_keys, 0)
        _clf = RandomForestClassifier(150)
        _clf.fit(x_train, y_train)
        print("train acc:", _clf.score(x_train, y_train))
        log_wt.write("train acc:", _clf.score(x_train, y_train))
        print("test acc:", _clf.score(x_test, y_test))
        log_wt.write("test acc:", _clf.score(x_test, y_test))
        log_wt.write("CLF", _clf)
        print()
        return _clf

    if is_fenceng == "fc":
        clf = fenceng()
        to_dict = clf.save()
        log_wt.write("MODEL_SAVE", to_dict)
    elif is_fenceng == "nofc":
        clf = not_fenceng()
        joblib.dump(clf, init_dfn.fn("nofc.mod"))
        log_wt.write("MODEL_SAVE", init_dfn.fn("nofc.mod"))
    else:
        clf = None

    # feature_importance(clf, nofc_x_keys)

    def imdc_fun(geo_fn, to_geo_fn, glcm_fn=None):
        print("geo_fn   :", geo_fn)
        print("to_geo_fn:", to_geo_fn)
        print("glcm_fn  :", glcm_fn)
        log_wt.write("GEO_FN:", geo_fn)
        log_wt.write("TO_GEO_FN:", to_geo_fn)
        log_wt.write("GLCM_FN:", glcm_fn)

        if is_fenceng == "nofc":
            imdc_keys = nofc_x_keys
        elif is_fenceng == "fc":
            imdc_keys = clf.get_keys()
            clf.init_imdc_keys(imdc_keys)
        else:
            print("imdc_fun is_fenceng == \"{}\"".format(is_fenceng))
            return

        log_wt.write("IMDC_KEYS:", imdc_keys)
        d = read_geo_raster(geo_fn, imdc_keys, glcm_fn)
        d[np.isnan(d)] = 0.0
        print(d.shape)

        imdc = np.zeros(d.shape[1:])
        jdt = Jdt(imdc.shape[0], "Imdc")
        jdt.start()
        for i in range(imdc.shape[0]):
            d_tmp = d[:, i, :].T
            y_tmp = clf.predict(d_tmp)
            imdc[i, :] = y_tmp
            jdt.add()
        jdt.end()

        to_fn = to_geo_fn
        gr = GDALRaster(geo_fn)
        gr.save(d=imdc, save_geo_raster_fn=to_fn, fmt="GTiff", dtype=gdal.GDT_Byte, options=["COMPRESS=PACKBITS"])
        code_colors = SHH_COLOR8
        tiffAddColorTable(to_fn, code_colors=code_colors)
        log_wt.write("CODE_COLORS:", code_colors)

        print()

        return {"GEO_FN": geo_fn, "TO_FN": to_fn}

    def imdc_fns():
        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        qd_to_fn = init_dfn.fn(r"qd_sh1_{0}.tif".format(is_fenceng))
        cd_to_fn = init_dfn.fn(r"cd_sh1_{0}.tif".format(is_fenceng))
        bj_to_fn = init_dfn.fn(r"bj_sh1_{0}.tif".format(is_fenceng))
        print(qd_to_fn, cd_to_fn, bj_to_fn, "", sep="\n")
        log_wt.write(qd_to_fn, cd_to_fn, bj_to_fn, "", sep="\n")
        imdc_dict = [
            imdc_fun(dfn.fn(r"QingDao\qd_sh2_1.tif"), qd_to_fn, dfn.fn(r"QingDao\glcm\qd_sh2_1_gray_envi_mean"), ),
            imdc_fun(dfn.fn(r"ChengDu\cd_sh2_1.tif"), cd_to_fn, dfn.fn(r"ChengDu\glcm\cd_sh2_1_gray_envi_mean"), ),
            imdc_fun(dfn.fn(r"BeiJing\bj_sh2_1.tif"), bj_to_fn, dfn.fn(r"BeiJing\glcm\bj_sh2_1_gray_envi_mean"), ),
        ]
        saveJson(imdc_dict, init_dfn.fn("imdc.json"))
        log_wt.write("IMDC_DICT", imdc_dict)
        with open(init_dfn.fn("imdc_gdaladdo.bat"), "w", encoding="utf-8") as f:
            f.write("gdaladdo \"{0}\"\n".format(qd_to_fn))
            f.write("gdaladdo \"{0}\"\n".format(cd_to_fn))
            f.write("gdaladdo \"{0}\"\n".format(bj_to_fn))

    # imdc_fns()


def tAccMLFC():
    # python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHMLFengCeng import tAccMLFC; tAccMLFC()"
    mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H120328nofc"
    mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H120401fc"
    mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H210341nofc"
    # mod_dirname = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc"
    print(mod_dirname)
    mod_dfn = DirFileName(mod_dirname)

    class t_acc_samples:

        def __init__(self):
            self.samples = []
            self.sdf = SRTDFColumnCal()

        def read_csv(self, csv_fn):
            self.sdf = SRTDFColumnCal()
            self.sdf.read_csv(csv_fn, is_auto_type=True)

        def init_train_data(self):
            csv_fn = mod_dfn.fn("train_data.csv")
            self.read_csv(csv_fn)
            return csv_fn

        def sampling(self, field_name, _imdc_fn, map_dict=None):
            o_rasters_sampling = GDALRastersSampling(_imdc_fn)

            def fit_func(line: dict):
                if "TEST" in line:
                    if line["TEST"] != 0:
                        return 0
                x, y = line["X"], line["Y"]
                d = o_rasters_sampling.sampling(x, y, 1, 1)
                if d is None:
                    return 0
                else:
                    d = d.ravel()
                cate = int(d[0])
                if map_dict is not None:
                    if cate in map_dict:
                        cate = map_dict[cate]
                return cate

            return self.sdf.fit(field_name, fit_func)

        def getCategory(self, c_name, map_dict=None):
            cate = self.sdf[c_name]
            if map_dict is not None:
                cate = [map_dict[k] if k in map_dict else k for k in cate]
            return cate

    sample_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\6\sh2_spl6_1_211.csv"
    spls = t_acc_samples()
    # spls.read_csv(sample_fn)
    sample_fn = spls.init_train_data()

    fn = getfilenamewithoutext(sample_fn)
    to_wt_fn = mod_dfn.fn(fn + "_tacc2.txt")
    print(to_wt_fn)
    to_wt = SRTWriteText(to_wt_fn)
    to_wt.write("MOD_DIRNAME: {0}\n".format(mod_dirname))
    to_wt.write("SAMPLE_FN: {0}\n".format(sample_fn))

    y1_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4}
    y1_map_dict = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8}
    to_wt.write("Y1_MAP_DICT: {0}\n".format(y1_map_dict))
    get_cname = "CATEGORY"
    to_wt.write("CNAME: {0}\n".format(get_cname))
    y0_map_dict = {11: 1, 21: 2, 31: 3, 41: 4, 12: 1, 22: 2, 32: 3, 42: 4}
    y0_map_dict = {11: 1, 21: 2, 31: 3, 41: 4, 12: 5, 22: 6, 32: 7, 42: 8}
    to_wt.write("Y0_MAP_DICT: {0}\n".format(y0_map_dict))
    y0 = spls.getCategory(get_cname, map_dict=y0_map_dict)

    cnames = [
        "IS", "VEG", "SOIL", "WAT",
        "IS_SH", "VEG_SH", "SOIL_SH", "WAT_SH",
    ]
    to_wt.write("CNAMES: {0}\n".format(cnames))

    imdc_fns = filterFileExt(mod_dirname, ".tif")
    to_wt.write("IMDC_FNS: {0}\n".format(fn))
    to_csv_fn = mod_dfn.fn(fn + "_tacc.csv")

    for imdc_fn in imdc_fns:
        fn = getfilenamewithoutext(imdc_fn)
        print(fn)
        to_wt.write("> {0}".format(fn))
        y1 = spls.sampling(fn, imdc_fn, y1_map_dict)
        y0_tmp, y1_tmp = [], []
        for i in range(len(spls.sdf)):
            if y1[i] != 0:
                y0_tmp.append(y0[i])
                y1_tmp.append(y1[i])
        cm = ConfusionMatrix(len(cnames), cnames)
        cm.addData(y0, y1)
        to_wt.write(cm.fmtCM(), "\n")
        print(cm.fmtCM())

    spls.sdf.toCSV(to_csv_fn)


def showFeatImp():
    data = [('OPT_var', 0.01576080067114541), ('OPT_dis', 0.016985082447471232), ('OPT_hom', 0.017616540976901025),
            ('OPT_asm', 0.017923906302913552), ('OPT_con', 0.018356980765199923), ('OPT_mean', 0.02191201533008487),
            ('OPT_ent', 0.022383564102921045), ('DE_VH', 0.030301017021598278), ('B3', 0.03451912048116178),
            ('DE_VV', 0.03620504625396766), ('AS_VH', 0.04421460000735884), ('mndwi', 0.04650880058323637),
            ('B2', 0.0466088179151466), ('B4', 0.050477364850457555), ('AS_VV', 0.05259860124221752),
            ('B12', 0.07709950041903517), ('B8', 0.08208353069186401), ('B11', 0.08227222530772832),
            ('ndwi', 0.13597161703821775), ('ndvi', 0.15020086759137327)]
    data = [('OPT_mean', 0.015812921221597682), ('OPT_asm', 0.016189493684199623), ('OPT_hom', 0.016671560648190056),
            ('OPT_var', 0.017104535091576435), ('OPT_dis', 0.017713110302578543), ('B11', 0.017746612066257408),
            ('B3', 0.019814833820484906), ('OPT_con', 0.02069767237997769), ('OPT_ent', 0.021570300884741992),
            ('B12', 0.023372443246070834), ('B4', 0.02478219558047086), ('B8', 0.02713507870334041),
            ('B2', 0.02770782061984352), ('mndwi', 0.04243660277886134), ('DE_VH', 0.06308501142811943),
            ('DE_VV', 0.07650474615203316), ('AS_VH', 0.10794083139911925), ('AS_VV', 0.12874449936121726),
            ('ndwi', 0.1469263044503744), ('ndvi', 0.1680434261809452)]

    # data.reverse()
    print(data)
    data = dict(data)
    plt.figure()
    plt.subplots_adjust(left=0.2)
    # 'Times New Roman'
    plt.xticks(fontproperties="Times New Roman", fontsize=16)
    plt.yticks(fontproperties="Times New Roman", fontsize=16)
    plt.barh(list(data.keys()), list(data.values()))
    plt.show()


def main():
    clf = joblib.load(r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc\fc_ws.mod")
    feature_importance(clf, ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndvi', 'ndwi', 'mndwi', 'AS_VV', 'AS_VH', 'DE_VV',
                             'DE_VH', 'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent', 'OPT_asm'])


def method_name1():
    # 计算精度
    df_fn = r"F:\ProjectSet\Shadow\Hierarchical\MLMods\20240308H215552fc\20240308H215552fc.xlsx"
    df = pd.read_excel(df_fn, sheet_name="Sheet2")
    df = df[df["FEN_CENG"] == 2]
    k0, k1 = "__CODE__", "FC_CATE"
    y0 = categoryMap(df[k0].values, SHHConfig.CATE_MAP_IS_8)
    y1 = categoryMap(df[k1].values, SHHConfig.CATE_MAP_IS_8)
    # y1 = df[k1].values
    cm = ConfusionMatrix(
        # class_names=SHHConfig.SHH_CNAMES[1:],
        class_names=SHHConfig.IS_CNAMES,
    )
    cm.addData(y0, y1)
    print(cm.fmtCM())
    print("OA Kappa")
    print(cm.OA(), cm.getKappa())


if __name__ == "__main__":
    showFeatImp()
