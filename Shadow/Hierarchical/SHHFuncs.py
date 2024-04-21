# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHFuncs.py
@Time    : 2024/3/4 13:14
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHFuncs
-----------------------------------------------------------------------------"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from SRTCodes.GDALRasterIO import GDALRaster, tiffAddColorTable
from SRTCodes.GDALUtils import replaceCategoryImage
from SRTCodes.Utils import changext, saveJson, SRTWriteText, timeStringNow, numberfilename, printList
from Shadow.Hierarchical.SHHRunMain import SHHModImSklearn


def rastersHist(*raster_fns, bins=256):
    to_dict = {}
    for raster_fn in raster_fns:
        if raster_fn in to_dict:
            continue
        to_dict[raster_fn] = {}
        gr = GDALRaster(raster_fn)
        print(raster_fn)
        for i in range(gr.n_channels):
            print(gr.names[i])
            to_dict[raster_fn][gr.names[i]] = {}
            d = gr.readGDALBand(i + 1)
            data, data_edge = np.histogram(d, bins=bins)
            to_dict[raster_fn]["DATA"] = data.tolist()
            to_dict[raster_fn]["DATA_EDGE"] = data_edge[:-1].tolist()
    return to_dict


class SHHReplaceCategoryImage:

    def __init__(self):
        self.name = "SHHReplaceCategoryImage"
        self.save_fn = r"F:\ProjectSet\Shadow\Hierarchical\ISModels\SHHReplaceCategoryImage.txt"
        self.wt = SRTWriteText(self.save_fn, mode="a")

    def VHL_IS(self, vhl_fn, is_fn, to_fn):
        o_map_dict = {1: 2, 2: 0, 3: 4}
        replace_map_dict = {1: 1, 2: 3}
        o_change = [0]
        color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (36, 36, 36)}
        replaceCategoryImage(
            o_geo_fn=vhl_fn, replace_geo_fn=is_fn, to_geo_fn=to_fn,
            o_map_dict=o_map_dict, replace_map_dict=replace_map_dict,
            o_change=o_change,
            color_table=color_table
        )
        to_shhrci_fn = changext(to_fn, "_rci.json")
        print("to_shhrci_fn:", to_shhrci_fn)
        to_dict = saveJson({
            "vhl_fn": vhl_fn,
            "is_fn": is_fn,
            "o_map_dict": o_map_dict,
            "replace_map_dict": {1: 3, 2: 1},
            "o_change": o_change,
            "color_table": color_table,
            "to_shhrci_fn": to_shhrci_fn
        }, to_shhrci_fn)
        self.to_wt(to_dict)

    def to_wt(self, to_dict):
        self.wt.write(">", timeStringNow())
        for i, k in enumerate(to_dict):
            self.wt.write("  + {0:<2d}".format(i + 1), k + ":", to_dict[k])
        self.wt.write()


class MLFuncs:

    def fit1Upate(self, csv_fn, to_csv_fn=None, is_tag_field_name="IS_TAG", category_field_name="CATEGORY", clf=None,
                  update_field_name=None):
        if clf is None:
            clf = RandomForestClassifier(150)
        if update_field_name is None:
            update_field_name = category_field_name
        if to_csv_fn is None:
            to_csv_fn = numberfilename(csv_fn, sep="_fit1Upate")
        shh_mis = SHHModImSklearn()
        df = pd.read_csv(csv_fn)
        df_fit = df[df[is_tag_field_name]]
        shh_mis.initPandas(df_fit)
        shh_mis.initCategoryField(field_name=category_field_name)
        fit_keys = [
            'B2', 'B3', 'B4', 'B8', 'B11', 'B12',
            'AS_VV', 'AS_VH', 'DE_VV', 'DE_VH',
            "NDVI", "NDWI", "MNDWI",
            'OPT_mean', 'OPT_var', 'OPT_hom', 'OPT_con', 'OPT_dis', 'OPT_ent',
        ]
        printList("fit_keys:", fit_keys)
        shh_mis.addNDVIFeatExt()
        shh_mis.addNDWIFeatExt()
        shh_mis.addMNDWIFeatExt()
        shh_mis.initXKeys(fit_keys)
        print("LEN X", len(shh_mis.x))
        shh_mis.initCLF(clf)
        shh_mis.train()
        shh_mis.scoreTrainCM()
        print("Train CM", shh_mis.train_cm.fmtCM(), sep="\n")

        to_category = shh_mis.predictDF(df)
        select_list = df[is_tag_field_name].values
        to_category[select_list] = df[select_list][category_field_name].values
        df[update_field_name] = to_category
        df.to_csv(to_csv_fn, index=False)
        print("to_csv_fn:", to_csv_fn)


def main():
    tiffAddColorTable(
        r"F:\ProjectSet\Shadow\Hierarchical\Mods\20240304H201631\model_epoch_86.pth_imdc1.tif",
        code_colors={0: (0, 0, 0, 0), 1: (0, 255, 0), 2: (220, 220, 220), 3: (60, 60, 60)}
    )
    pass


if __name__ == "__main__":
    main()
