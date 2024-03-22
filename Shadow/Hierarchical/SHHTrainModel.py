# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHTrainModel.py
@Time    : 2024/3/2 19:31
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHTrainModel
-----------------------------------------------------------------------------"""
import os
import sys
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.models import VisionTransformer

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.Utils import writeCSVLine, DirFileName, mkdir, getfilenme, Jdt
from SRTCodes.GDALRasterIO import tiffAddColorTable
from Shadow.Hierarchical.SHHRunMain import SHHMainInit
from Shadow.Hierarchical.ShadowHSample import loadSHHSamples, ShadowHierarchicalSampleCollection


class SHHModel_TO3(nn.Module):

    def __init__(self):
        super(SHHModel_TO3, self).__init__()

        self.vit = VisionTransformer(
            image_size=12,
            patch_size=3,
            num_layers=12,
            num_heads=12,
            hidden_dim=120,
            mlp_dim=600,
            dropout=0.2,
            attention_dropout=0.2,
            num_classes=3,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs=None,
        )

    def forward(self, x):
        x = self.vit(x)
        return x


class SHHModel_TO3_2(nn.Module):

    def __init__(self):
        super(SHHModel_TO3_2, self).__init__()

        def cbr(in_c, out_c, ks, stride, padding, ):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, ks, stride, padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        self.cbr1 = cbr(6, 32, (3, 3), 1, 1)
        self.cbr2 = cbr(32, 64, (3, 3), 1, 1)
        self.max_pooling1 = nn.MaxPool2d((2, 2), 2)
        self.cbr3 = cbr(64, 128, (3, 3), 1, 1)
        self.cbr4 = cbr(128, 256, (3, 3), 1, 1)
        self.max_pooling2 = nn.MaxPool2d((2, 2), 2)

        self.fc1 = nn.Linear(256, 1000)
        self.fc2 = nn.Linear(1000, 256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.max_pooling1(x)

        x = self.cbr3(x)
        x = self.cbr4(x)
        x = self.max_pooling2(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def shhTo3_dealData(x, y=None):
    x = x[6:12]
    x = x / 2000
    if y is not None:
        y = y - 1
        return x, y
    return x


class SHHTM_Dataset(Dataset):

    def __init__(self, shh_sc: ShadowHierarchicalSampleCollection):
        self.shh_sc: ShadowHierarchicalSampleCollection = shh_sc
        self.shh_sc.ndc.__init__(3, (13, 13), (21, 21))
        # self.shh_sc.initVegHighLowCategoryCollMap()

    def __getitem__(self, index) -> T_co:
        x = self.shh_sc.data(index, is_center=True)
        y = self.shh_sc.category(index)
        x, y = shhTo3_dealData(x, y)
        x = x.astype("float32")
        return x, y

    def __len__(self):
        return len(self.shh_sc)


class SHHModel_TO3_ptt(PytorchCategoryTraining):

    def __init__(self, model_dir=None, model_name="model", n_category=2, category_names=None, epochs=10, device=None,
                 n_test=100):
        super().__init__(model_dir=model_dir, model_name=model_name, n_category=n_category,
                         category_names=category_names, epochs=epochs, device=device, n_test=n_test)

    def logisticToCategory(self, logts):
        logts = torch.argmax(logts, dim=1)
        return logts

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):
        self._initTrain()
        self._printModel()
        self._log.saveHeader()

        for epoch in range(self.epochs):

            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.float(), y.long()

                self.model.train()
                logts = self.model(x)

                self.loss = self.criterion(logts, y)
                self.loss = self.lossDeal(self.loss)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                if self.test_loader is not None:
                    if batchix % self.n_test == 0:
                        self.testAccuracy()
                        modname = self.log(batchix, epoch)
                        if batch_save:
                            self.saveModel(modname)

            print("-" * 80)
            self.testAccuracy()
            self.log(-1, epoch)
            modname = self.model_name + "_epoch_{0}.pth".format(epoch)
            print("*" * 80)

            if epoch_save:
                self.saveModel(modname)

    def testAccuracy(self):
        self.test_cm.clear()
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                x = x.float()
                y = y.numpy()
                y = y + 1

                logts = self.model(x)
                y1 = self.logisticToCategory(logts)
                y1 = y1 + 1

                y1 = y1.cpu().numpy()
                self.test_cm.addData(y, y1)
        self.model.train()
        return self.test_cm.OA()


class SHHModel_TO3_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(SHHModel_TO3_GDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]
        x = torch.from_numpy(x)
        x = x.float()
        x = x.to(self.device)
        y = torch.zeros((n), dtype=torch.float)
        y = y.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n, self.number_pred):
                y_temp = self.model(x[i:i + self.number_pred, :])
                y_temp = torch.argmax(y_temp, dim=1)
                y[i:i + self.number_pred] = y_temp

        y = y + 1
        y = y.cpu().numpy()

        return y

    def preDeal(self, row, column_start, column_end):
        d_row = self.d[:, row, column_start:column_end]
        ndvi = (d_row[3, :] - d_row[2, :]) / (d_row[3, :] + d_row[2, :])
        # np.ones(d_row.shape[1], dtype="bool")
        return ndvi < 0.5


class SHHModel_TO3_Main(SHHMainInit):

    def __init__(self):
        super().__init__()

        self.n_category = 3
        self.category_names = ["VEG", "HIGH", "LOW"]
        self.epochs = 200
        self.device = "cuda:0"
        self.n_test = 10
        self.test_ds = None
        self.train_ds = None
        self.win_size = 12
        self.mod = SHHModel_TO3()

        self.geo_raster = self.qd_geo_raster

    def train(self):
        print("raster_fn :", self.geo_raster)
        print("spl_size  :", self.win_size)

        batch_size = 128

        print("Export Data")
        self.loadSHHSamplesDS()

        pytorch_training = SHHModel_TO3_ptt(
            n_category=self.n_category,
            model_dir=self.model_dir,
            category_names=self.category_names,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test
        )

        pytorch_training.trainLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        pytorch_training.testLoader(self.test_ds, batch_size=batch_size, shuffle=False)
        pytorch_training.addModel(self.mod)
        pytorch_training.addCriterion(nn.CrossEntropyLoss())
        pytorch_training.addOptimizer(lr=0.001, eps=0.00001)

        print("model_dir", pytorch_training.model_dir)
        self.mod_dirname = pytorch_training.model_dir
        self.saveCodeFile(pytorch_training.model_dir, __file__)
        self.saveCodeFile(pytorch_training.model_dir)
        # save_fn = os.path.join(pytorch_training.model_dir, "save.txt")
        # writeTexts(save_fn, "spl_size  :", self.win_size, mode="a", end="\n")
        pytorch_training.train()

    def loadSHHSamplesDS(self):
        shh_sc_train, shh_sc_test = loadSHHSamples("qd_sample4[21,21]")
        self.train_ds = SHHTM_Dataset(shh_sc_train)
        self.test_ds = SHHTM_Dataset(shh_sc_test)
        # self.test_ds = SHHTM_Dataset(shh_sc_train)
        # self.train_ds = SHHTM_Dataset(shh_sc_test)

    def samplesCategory(self, mod_fn=None):
        mod_fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\20240305H101932\model_epoch_198.pth"
        if mod_fn is None:
            mod_fn = sys.argv[1]
        self.mod_dirname = os.path.dirname(mod_fn)
        spl_csv_fn = mod_fn + "_spl1.csv"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        self.loadSHHSamplesDS()
        ds = self.train_ds
        ds: SHHTM_Dataset

        print("mod_dirname:", self.mod_dirname)
        print("mod_fn     :", mod_fn)
        print("spl_csv_fn :", spl_csv_fn)

        self.mod.eval()
        jdt = Jdt(len(ds), "SHHModel_TO3_Main::samplesCategory")
        jdt.start()
        for i, (x, y) in enumerate(ds):
            jdt.add()
            x = torch.from_numpy(x)
            x = x.float().to(self.device)
            x = torch.unsqueeze(x, 0)
            logts = self.mod(x)
            y1 = torch.argmax(logts, dim=1)
            y1 = y1.cpu().item() + 1
            ds.shh_sc.setField(i, "Y_PRED", int(y1))

        jdt.end()

        ds.shh_sc.toCSV(spl_csv_fn)

    def accCal(self, model, test_loader, device="cuda"):
        model.eval()
        with torch.no_grad():
            total_number = 0
            n_true = 0
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                x, y = x.float(), y.long()
                logts = model(x)
                y_pred = torch.argmax(logts, dim=1)
                total_number += len(y_pred)
                n_true += torch.sum(y == y_pred).item()
                pass
            acc = n_true * 1.0 / total_number
        model.train()
        return acc * 100

    def imdcOne(self, mod_fn=None, to_imdc_name=None):
        if mod_fn is None:
            mod_fn = sys.argv[1]
        if to_imdc_name is None:
            to_imdc_name = sys.argv[2]
        # mod_fn = r"F:\ProjectSet\Shadow\Hierarchical\Mods\20240305H101242\model_epoch_66.pth"
        # to_imdc_name= 1
        self.mod_dirname = os.path.dirname(mod_fn)
        imdc_fn = mod_fn + "_imdc2.tif".format(to_imdc_name)
        grp = SHHModel_TO3_GDALRasterPrediction(self.geo_raster)

        print("mod_dirname:", self.mod_dirname)
        print("imdc_fn    :", imdc_fn)
        print("mod_fn     :", mod_fn)

        grp.is_category = True
        np_type = "int8"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                spl_size=[self.win_size, self.win_size],
                row_start=self.win_size + 6, row_end=-(self.win_size + 6),
                column_start=self.win_size + 6, column_end=-(self.win_size + 6),
                n_one_t=15000, data_deal=shhTo3_dealData)
        tiffAddColorTable(imdc_fn, code_colors={0: (0, 255, 0), 1: (0, 255, 0), 2: (220, 220, 220), 3: (60, 60, 60)})


def main():
    SHHModel_TO3_Main().train()

    # mod = SHHModel_TO3_2()
    # x = torch.rand(8, 6, 7, 7)
    # out_x = mod(x)
    return


if __name__ == "__main__":
    """
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHTrainModel import SHHModel_TO3_Main; SHHModel_TO3_Main().imdcOne()"
python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHHTrainModel import SHHModel_TO3_Main; SHHModel_TO3_Main().samplesCategory()"
    """
    main()
