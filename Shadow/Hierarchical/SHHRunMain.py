# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHRunMain.py
@Time    : 2024/3/7 16:12
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHRunMain
-----------------------------------------------------------------------------"""
import os
import sys
from datetime import datetime

import torch
from torch import nn

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.Utils import mkdir, getfilenme, writeCSVLine, DirFileName, Jdt
from SRTCodes.GDALRasterIO import tiffAddColorTable


class SHHMainInit:

    def __init__(self):
        self.this_dirname = mkdir(r"F:\ProjectSet\Shadow\Hierarchical")
        self.model_dir = mkdir(os.path.join(self.this_dirname, "Mods"))
        self.code_text_filename = None
        self.acc_text_filename = None
        self.formatted_time = None
        self.mod_dirname = None
        self.name = None
        self.device = "cuda"
        self.test_ds = None
        self.train_ds = None

        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\Images")
        self.qd_geo_raster = dfn.fn("qd_sh2_1.tif")
        self.bj_geo_raster = dfn.fn("bj_sh2_1.tif")
        self.cd_geo_raster = dfn.fn("cd_sh2_1.tif")
        self.qd_geo_raster1 = r"F:\ProjectSet\Shadow\Hierarchical\Images\test\qd_sh2_1_t1.tif"
        self.qd_geo_raster2 = r"F:\ProjectSet\Shadow\Hierarchical\Images\test\qd_sh2_1_t2.tif"
        self.qd_geo_raster3 = r"F:\ProjectSet\Shadow\Hierarchical\Images\test\qd_sh2_1_t3.tif"
        self.geo_raster = None

        self.n_category = 3
        self.category_names = ["VEG", "HIGH", "LOW"]
        self.epochs = 200
        self.n_test = 10
        self.win_size = 12
        self.batch_size = 128
        self.mod = None

        self.geo_raster = self.qd_geo_raster

    def timeDirName(self):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        to_dir = os.path.join(self.this_dirname, formatted_time)
        if not os.path.isdir(to_dir):
            os.mkdir(to_dir)
        return formatted_time, to_dir

    def saveCodeFile(self, to_dir=None, code_fn=None):
        if code_fn is None:
            code_fn = self.code_text_filename
        fn = getfilenme(code_fn)
        to_fn = os.path.join(to_dir, fn)
        with open(to_fn, "w", encoding="utf-8") as f:
            with open(code_fn, "r", encoding="utf-8") as fr:
                text = fr.read()
            f.write(text)

    def writeAccText(self, *line):
        writeCSVLine(self.acc_text_filename, list(line))

    def getModelFileName(self, epoch):
        return os.path.join(self.mod_dirname, "{0}_{1}.pth".format(self.formatted_time, epoch))

    def loadSamplesDS(self, *args, **kwargs):
        return None

    def train(self, pct_class=PytorchCategoryTraining, *args, **kwargs):
        print("raster_fn :", self.geo_raster)
        print("spl_size  :", self.win_size)
        print("Export Data")
        self.loadSamplesDS()
        print("Number of train samples:", len(self.train_ds))
        print("Number of train samples:", len(self.test_ds))

        pytorch_training = pct_class(
            n_category=self.n_category,
            model_dir=self.model_dir,
            category_names=self.category_names,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test
        )

        pytorch_training.trainLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        pytorch_training.testLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
        pytorch_training.addModel(self.mod)
        pytorch_training.addCriterion(nn.CrossEntropyLoss())
        pytorch_training.addOptimizer(lr=0.0005, eps=0.00001)

        print("model_dir", pytorch_training.model_dir)
        self.mod_dirname = pytorch_training.model_dir
        self.saveCodeFile(pytorch_training.model_dir)
        # save_fn = os.path.join(pytorch_training.model_dir, "save.txt")
        # writeTexts(save_fn, "spl_size  :", self.win_size, mode="a", end="\n")
        pytorch_training.train()

    def samplesCategory(self, mod_fn=None, logit_cate_func=None, to_csv_fn=None, ds=None, *args, **kwargs):
        if mod_fn is None:
            mod_fn = sys.argv[1]
        if to_csv_fn is None:
            to_csv_fn = sys.argv[2]
        spl_csv_fn = mod_fn + "_imdc2.tif".format(to_csv_fn)
        if ds is None:
            self.loadSamplesDS()
            ds = self.train_ds

        self.mod_dirname = os.path.dirname(mod_fn)
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        self.mod.eval()

        print("mod_dirname:", self.mod_dirname)
        print("mod_fn     :", mod_fn)
        print("spl_csv_fn :", spl_csv_fn)

        jdt = Jdt(len(ds), "SHHModel_TO3_Main::samplesCategory")
        jdt.start()
        for i, (x, y) in enumerate(ds):
            jdt.add()
            x = torch.from_numpy(x)
            x = x.float().to(self.device)
            x = torch.unsqueeze(x, 0)
            logts = self.mod(x)
            if logit_cate_func is not None:
                y1 = logit_cate_func(logts)
            else:
                y1 = torch.argmax(logts, dim=1) + 1
            y1 = y1.cpu().item()
            ds.shh_sc.setField(i, "Y_PRED", int(y1))
        jdt.end()
        ds.shh_sc.toCSV(spl_csv_fn)

    def imdcOne(self, mod_fn=None, to_imdc_name=None, grp: GDALRasterPrediction = None,
                data_deal=None, code_colors=None,
                *args, **kwargs):
        """ code_colors {0: (0, 0, 0), **} """
        if code_colors is None:
            code_colors = {0: (0, 255, 0), 1: (0, 255, 0), 2: (220, 220, 220), 3: (60, 60, 60)}
        if mod_fn is None:
            mod_fn = sys.argv[1]
        if to_imdc_name is None:
            to_imdc_name = sys.argv[2]
        imdc_fn = mod_fn + "_imdc2.tif".format(to_imdc_name)

        self.mod_dirname = os.path.dirname(mod_fn)
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        self.mod.eval()

        print("mod_dirname:", self.mod_dirname)
        print("imdc_fn    :", imdc_fn)
        print("mod_fn     :", mod_fn)

        grp.is_category = True
        np_type = "int8"

        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                spl_size=[self.win_size, self.win_size],
                row_start=self.win_size + 6, row_end=-(self.win_size + 6),
                column_start=self.win_size + 6, column_end=-(self.win_size + 6),
                n_one_t=15000, data_deal=data_deal)
        tiffAddColorTable(imdc_fn, code_colors=code_colors)


def main():
    pass


if __name__ == "__main__":
    main()
