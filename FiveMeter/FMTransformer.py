# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : FMTransformer.py
@Time    : 2024/1/21 14:16
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of FMTransformer
-----------------------------------------------------------------------------"""
import os
from datetime import datetime
from functools import partial

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import VisionTransformer

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.Utils import readJson, writeCSVLine


def tensorCenter(tensor: torch.Tensor, size):
    row, column = int(tensor.shape[2] / 2.0), int(tensor.shape[3] / 2.0)
    size_row, size_column = int(size[0] / 2.0), int(size[1] / 2.0)
    tensor_out = tensor[:, :, row - size_row:row + size_row, column - size_column:column + size_column]
    return tensor_out


class FMTModel(nn.Module):

    def __init__(self):
        super(FMTModel, self).__init__()

        self.vit = VisionTransformer(
            image_size=12,
            patch_size=2,
            num_layers=12,
            num_heads=12,
            hidden_dim=120,
            mlp_dim=600,
            dropout=0.2,
            attention_dropout=0.2,
            num_classes=2,
            representation_size=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            conv_stem_configs=None,
        )

    def forward(self, x):
        x = tensorCenter(x, (12, 12))
        x = self.vit(x)
        return x


class FMDataSet(Dataset):

    def __init__(self, np_filename, json_filename):
        super(FMDataSet, self).__init__()

        self.np_fn = np_filename
        self.json_fn = json_filename
        self.data = None
        self.json_data = {}

        self.readData()

    def readData(self):
        self.data = np.load(self.np_fn)
        self.json_data = readJson(self.json_fn)
        # self.labels = [json_d["Name"] for json_d in self.json_data]

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        x = self.data[index]
        x = x / 1600
        y = self.json_data[index]["Name"]
        return x, y


class FMT_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(FMT_GDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]

        x = x.astype("float32")
        x = torch.from_numpy(x)
        x = x.to(self.device)
        y = torch.zeros((n, 1), dtype=torch.float)
        y = y.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n, self.number_pred):
                y_temp = self.model(x[i:i + self.number_pred, :])
                y[i:i + self.number_pred, :] = y_temp
            y = functional.sigmoid(y)
        y = y.cpu().numpy()
        y = y.T[0]
        if self.is_category:
            y = (y > 0.5) * 1

        return y

    def preDeal(self, row, column_start, column_end):
        d_row = self.d[:, row, column_start:column_end]
        return np.ones(d_row.shape[1], dtype="bool")


class FMTMain:

    def __init__(self):
        self.name = "FMTMain"
        self.init_dir = r"G:\FM\Mods"
        formatted_time, to_dir = self.timeDirName()
        self.acc_text_filename = os.path.join(to_dir, "{0}_acc.csv".format(formatted_time))
        self.code_text_filename = os.path.join(to_dir, "{0}_code.py".format(formatted_time))
        self.mod_dirname = to_dir
        print(self.mod_dirname)
        self.formatted_time = formatted_time
        self.saveCodeFile()

    def timeDirName(self):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        to_dir = os.path.join(self.init_dir, formatted_time)
        if not os.path.isdir(to_dir):
            os.mkdir(to_dir)
        return formatted_time, to_dir

    def saveCodeFile(self):
        with open(self.code_text_filename, "w", encoding="utf-8") as f:
            with open(__file__, "r", encoding="utf-8") as fr:
                text = fr.read()
            f.write(text)

    def writeAccText(self, *line):
        writeCSVLine(self.acc_text_filename, list(line))

    def getModelFileName(self, epoch):
        return os.path.join(self.mod_dirname, "{0}_{1}.pth".format(self.formatted_time, epoch))

    def train(self):
        epochs = 100
        batch_size = 256
        n_test = 2
        n_save = 10
        device = "cuda"

        print("Export Test Data")
        test_dataset = FMDataSet(
            np_filename=r"L:\LMLY\test\Data_1.npy",
            json_filename=r"L:\LMLY\test\Data_y.json",
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("Export Train Data")
        train_dataset = FMDataSet(
            np_filename=r"L:\LMLY\train\lm_crf_data.npy",
            json_filename=r"L:\LMLY\train\lm_crf_pro.json",
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = FMTModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 学习率

        self.writeAccText("Epoch", "Batch", "Accuracy", "Loss")
        for epoch in range(epochs):
            loss = None

            for batchix, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                x, y = x.float(), y.long()

                logts = model(x)
                loss = criterion(logts, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if test_loader is not None:
                    if batchix % n_test == 0:
                        acc = self.accCal(model, test_loader)
                        self.writeAccText(epoch, batchix, acc, loss.item())
                        print("Epoch:{0:2d} Batch:{1:3d} Loss:{3:10.6f} Acc:{2:6.2f}%".format(
                            epoch, batchix, acc, loss.item()))

                if n_save > 0:
                    if batchix % n_save == 0:
                        mod_fn = self.getModelFileName(epoch)
                        torch.save(model.state_dict(), mod_fn)

            acc = self.accCal(model, test_loader)
            print("Epoch:{0:2d} Acc:{1:6.2f}% Loss:{2:10.6f} ".format(epoch, acc, loss.item()))
            self.writeAccText(epoch, -1, acc, loss.item())
            print("-" * 60)
            # 测试
            mod_fn = self.getModelFileName(epoch)
            torch.save(model.state_dict(), mod_fn)

    def trainOCC(self):
        epochs = 100
        batch_size = 256
        n_test = 2
        n_save = 10
        device = "cuda"

        print("Export Test Data")
        test_dataset = FMDataSet(
            np_filename=r"L:\LMLY\test\Data_1.npy",
            json_filename=r"L:\LMLY\test\Data_y.json",
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print("Export Train Data")
        train_dataset = FMDataSet(
            np_filename=r"L:\LMLY\train\lm_crf_data.npy",
            json_filename=r"L:\LMLY\train\lm_crf_pro.json",
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = FMTModel().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 学习率

        self.writeAccText("Epoch", "Batch", "Accuracy", "Loss")
        for epoch in range(epochs):
            loss = None

            for batchix, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                x, y = x.float(), y.long()

                logts = model(x)
                loss = criterion(logts, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if test_loader is not None:
                    if batchix % n_test == 0:
                        acc = self.accCal(model, test_loader)
                        self.writeAccText(epoch, batchix, acc, loss.item())
                        print("Epoch:{0:2d} Batch:{1:3d} Loss:{3:10.6f} Acc:{2:6.2f}%".format(
                            epoch, batchix, acc, loss.item()))

                if n_save > 0:
                    if batchix % n_save == 0:
                        mod_fn = self.getModelFileName(epoch)
                        torch.save(model.state_dict(), mod_fn)

            acc = self.accCal(model, test_loader)
            print("Epoch:{0:2d} Acc:{1:6.2f}% Loss:{2:10.6f} ".format(epoch, acc, loss.item()))
            self.writeAccText(epoch, -1, acc, loss.item())
            print("-" * 60)
            # 测试
            mod_fn = self.getModelFileName(epoch)
            torch.save(model.state_dict(), mod_fn)

    def accCal(self, model, test_loader, device="cuda"):
        with torch.no_grad():
            total_number = 0
            n_true = 0
            for i, (x, y) in enumerate(test_loader):
                x, y = x.to(device), y.to(device)
                logts = model(x)
                y_pred = torch.argmax(logts, dim=1)
                total_number += len(y_pred)
                n_true += torch.sum(y == y_pred).item()
                pass
            acc = n_true * 1.0 / total_number
        return acc * 100


def main():
    # mod = VisionTransformer(
    #     image_size=12,
    #     patch_size=2,
    #     num_layers=6,
    #     num_heads=4,
    #     hidden_dim=30,
    #     mlp_dim=300,
    #     dropout=0.0,
    #     attention_dropout=0.0,
    #     num_classes=2,
    #     representation_size=None,
    #     norm_layer=partial(nn.LayerNorm, eps=1e-6),
    #     conv_stem_configs=None,
    # )
    # x = torch.rand((32, 4, 12, 12))
    # x = mod(x)
    # torch.save(mod.state_dict(), r"F:\Week\20240121\Data\mod.pth")
    # print(mod)
    # print(x.shape)
    # nn_list = dir(nn)
    # for d in nn_list:
    #     print(d)
    # d =readJson(r"L:\LMLY\train\lm_crf_pro.json")
    # df = pd.DataFrame(d)
    # df.to_csv(r"L:\LMLY\train\lm_crf_pro.csv", index=False)

    fmtm = FMTMain()
    fmtm.train()

    # mod = FMTModel()
    # print(mod)
    # torch.save(mod.state_dict(), r"G:\FM\Mods\20240121213954_0.pth")

    # mod.load_state_dict(torch.load(r"G:\FM\Mods\20240121213954_0.pth"))


if __name__ == "__main__":
    main()
