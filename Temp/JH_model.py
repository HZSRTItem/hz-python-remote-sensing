# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : JH_model.py
@Time    : 2024/3/24 20:55
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of JH_model
-----------------------------------------------------------------------------"""
import os
import random

import numpy as np
import torch
from osgeo import gdal
from torch import nn
from torch.utils.data import DataLoader, Dataset

from SRTCodes.ModelTraining import ConfusionMatrix
from SRTCodes.Utils import timeDirName, DirFileName, writeCSVLine, copyFile, FN, changext, Jdt, readLines


class JHModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def func1(in_c, out_c, kernel_size=3):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, 1, 1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            )

        self.layer1 = func1(3, 16)
        self.layer2 = func1(16, 32)
        self.pooling1 = nn.MaxPool2d(2, 2)
        self.layer3 = func1(32, 64)
        self.layer4 = func1(64, 128)
        self.pooling2 = nn.MaxPool2d(2, 2)
        self.layer5 = func1(128, 128, 1)

        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pooling1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling2(x)
        x = self.layer5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class YuYiFenGe(nn.Module):

    def __init__(self):
        super(YuYiFenGe, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def readData(fn):
    ds = gdal.Open(fn)
    data: np.ndarray = ds.ReadAsArray()
    return data


def sampling(image_data, label_data, category, n, win_row, win_column):
    data_row, data_column = label_data.shape
    row, column = np.where(label_data == category)
    select_list = [i for i in range(len(row))]
    random.shuffle(select_list)
    select_list = select_list[:n]
    row, column = row[select_list], column[select_list]
    data_list = []
    label_list = []
    win_row_2, win_column_2 = int(win_row / 2), int(win_column / 2)
    for i in range(len(select_list)):
        r, c = row[i], column[i]
        if ((r - win_row) < 2) or ((c - win_column) < 2) or ((data_row - r) < (win_row + 2)) \
                or ((data_column - c) < (win_column + 2)):
            continue
        data_list.append([image_data[:, r - win_row_2:r + win_row_2 + 1, c - win_column_2:c + win_column_2 + 1]])
        label_list.append(category)
    return np.concatenate(data_list), np.array(label_list)


class JHDataset(Dataset):

    def __init__(self, data, label):
        super(JHDataset, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        x, y = self.data[item], self.label[item] - 1

        return x, y


def writeTiffDS(data, filename):
    driver = gdal.GetDriverByName("GTiff")  # 申请空间
    dst_ds = driver.Create(filename, data.shape[2], data.shape[1], 4, gdal.GDT_Byte, [])  # 列数 行数 波段数
    for i in range(4):
        band: gdal.Band = dst_ds.GetRasterBand(i + 1)
        band.WriteArray(data[i])

    # band: gdal.Band = dst_ds.GetRasterBand(1)
    # code_colors = {1: (0, 255, 0), 2: (255, 255, 0), 3: (0, 0, 255)}
    # color_table = gdal.ColorTable()
    # for c_code, color in code_colors.items():
    #     color_table.SetColorEntry(c_code, color)
    # band.WriteArray(data)
    # band.SetColorTable(color_table)

    return dst_ds


def writePng(data, filename):
    data = data.astype("int8")
    data = data * 252

    tif_fn = changext(filename, ".tif")
    tif_ds = writeTiffDS(data, tif_fn)

    png_fn = changext(filename, ".png")
    driver = gdal.GetDriverByName("PNG")
    dst_ds = driver.CreateCopy(png_fn, tif_ds)

    del tif_ds
    del dst_ds

    os.remove(tif_fn)


def loadData():
    def func1():
        image_data = readData(r"F:\Week\20240324\Data\Part_Huangdao.jpg")
        print("image_data.shape", image_data.shape)
        label_data = readData(r"F:\Week\20240324\Data\Part_Huangdao.png")
        print("label_data.shape", label_data.shape)

        # plt.imshow(image_data.transpose((1,2,0)))
        # plt.show()
        # plt.imshow(label_data)
        # plt.show()

        def get_data(n):
            data1, label1 = sampling(image_data, label_data, 1, n, 9, 9)
            data2, label2 = sampling(image_data, label_data, 2, n, 9, 9)
            data3, label3 = sampling(image_data, label_data, 3, n, 9, 9)

            data = np.concatenate([data1, data2, data3])
            label = np.concatenate([label1, label2, label3])

            return data, label

        train_data, train_label = get_data(10000)
        test_data, test_label = get_data(1000)

        train_ds = JHDataset(train_data, train_label)
        test_ds = JHDataset(test_data, test_label)

        return train_ds, test_ds

    def func2():

        dfn = DirFileName(r"F:\Week\20240414\Data\Desktop\crop")
        images_dfn = DirFileName(dfn.fn("images"))
        labels_dfn = DirFileName(dfn.fn("labels"))

        def data_deal2(x):
            return x / 255.0

        def load_data(txt_fn):
            lines = readLines(txt_fn)
            _datas = [data_deal2(readData(images_dfn.fn(fn + ".tif"))) for fn in lines]
            _labels = [readData(labels_dfn.fn(fn + ".tif")) for fn in lines]
            return _datas, _labels

        train_datas, train_labels = load_data(dfn.fn("train.txt"))
        test_datas, test_labels = load_data(dfn.fn("val.txt"))
        train_ds = JHDataset(train_datas, train_labels)
        test_ds = JHDataset(test_datas, test_labels)
        return train_ds, test_ds

    return func2()


def main():
    # mod = JHModel()
    # x = torch.rand((32, 3, 9, 9))
    # logt = mod(x)
    train_1()
    return


def predict():
    def func1():
        image_data = readData(r"F:\Week\20240324\Data\Part_Huangdao.jpg")
        image_data = image_data.astype("float32")
        image_data = image_data / 255
        mod_fn = r"F:\Week\20240324\Data\20240324H222252\model_198_923.pth"
        device = "cuda"
        mod = JHModel()
        mod.load_state_dict(torch.load(mod_fn))
        mod.to(device)

        imdc = np.zeros(image_data.shape[1:])
        row_start, row_end, = 10, imdc.shape[0] - 10
        column_start, column_end, = 10, imdc.shape[1] - 10
        win_row = 9
        win_column = 9
        win_row_2, win_column_2 = int(win_row / 2), int(win_column / 2)

        jdt = Jdt(row_end - row_start, "predict").start(is_jdt=True)
        for i in range(row_start, row_end):
            data_line = []

            for j in range(column_start, column_end):
                r, c = i, j
                data_line.append(
                    [image_data[:, r - win_row_2:r + win_row_2 + 1, c - win_column_2:c + win_column_2 + 1]])

            data_line = np.concatenate(data_line)
            data_line = torch.from_numpy(data_line).to(device)

            logit = mod(data_line)
            y = torch.argmax(logit, dim=1)
            y = y.cpu().numpy()
            imdc[i, column_start: column_end] = y + 1

            jdt.add(is_jdt=True)
        jdt.end(is_jdt=True)

        np.save(changext(mod_fn, "_imdcdata.npy"), imdc.astype("int8"))
        to_fn = changext(mod_fn, "_imdc_tif.tif")
        print(to_fn)
        to_data = np.concatenate([image_data * 255, [imdc]])
        writeTiffDS(to_data.astype("int8"), to_fn)
        # plt.imshow(imdc)
        # plt.savefig(changext(mod_fn, "_imdc.png"))
        # plt.show()

    def fucn2():
        data = np.load(r"F:\Week\20240324\Data\20240324H222252\model_198_923_imdcdata.npy")
        filename = r"F:\Week\20240324\Data\20240324H222252\model_198_923_imdc_tif.tif"

        driver = gdal.GetDriverByName("GTiff")  # 申请空间
        dst_ds = driver.Create(filename, data.shape[1], data.shape[0], 1, gdal.GDT_Byte, [])  # 列数 行数 波段数
        band: gdal.Band = dst_ds.GetRasterBand(1)
        band.WriteArray(data)
        return dst_ds

    def func3():
        dfn = DirFileName(r"F:\Week\20240414\Data\Desktop\crop")
        images_dfn = DirFileName(dfn.fn("images"))
        labels_dfn = DirFileName(dfn.fn("labels"))

        def data_deal2(x):
            return x / 255.0

        def load_data(txt_fn):
            lines = readLines(txt_fn)
            _datas = [data_deal2(readData(images_dfn.fn(fn + ".tif"))) for fn in lines]
            _labels = [readData(labels_dfn.fn(fn + ".tif")) for fn in lines]
            return _datas, _labels, lines

        train_datas, train_labels, train_lines = load_data(dfn.fn("train.txt"))
        test_datas, test_labels, test_lines =  load_data(dfn.fn("val.txt"))

        device = "cuda"
        model = YuYiFenGe().to(device)
        mod_fn = r"F:\Week\20240414\Data\Model\20240410H193316\model_10_49.pth"
        model.load_state_dict(torch.load(mod_fn))

        to_dirname = mod_fn + "_imdc"
        if not os.path.isdir(to_dirname):
            os.mkdir(to_dirname)

        dfn = DirFileName(to_dirname)

        code_colors = {1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255), 4: (0, 0, 0)}

        def predict_func(_datas, _lines):
            jdt = Jdt(len(_datas), "predict_func").start()
            for i, data in enumerate(_datas):
                data = np.expand_dims(data, axis=0)
                data = torch.from_numpy(data).to(device).float()
                logts = model(data)
                y = torch.argmax(logts, dim=1)
                y = y.cpu().numpy() + 1
                y = y[0]

                method_name(_lines[i], y, dfn)

                jdt.add()
            jdt.end()

        def method_name(_lines_i, y, _dfn):
            color_table = gdal.ColorTable()
            for c_code, color in code_colors.items():
                color_table.SetColorEntry(c_code, color)
            filename = _dfn.fn(_lines_i + ".tif")
            driver = gdal.GetDriverByName("GTiff")  # 申请空间
            dst_ds = driver.Create(filename, y.shape[1], y.shape[0], 1, gdal.GDT_Byte, [])  # 列数 行数 波段数
            band: gdal.Band = dst_ds.GetRasterBand(1)
            band.WriteArray(y)
            band.SetColorTable(color_table)

        def predict_func_labels(_labels, _lines):
            _dfn = DirFileName(r"F:\Week\20240414\Data\labels_show")
            for i, data in enumerate(_labels):
                method_name(_lines[i], data, _dfn)

        predict_func(train_datas, train_lines)
        predict_func(test_datas, test_lines)
        predict_func_labels(train_labels, train_lines)
        predict_func_labels(test_labels, test_lines)

    func3()


def train_1():
    def data_deal(x, y=None):
        out_x = np.zeros((10, 3, 3))
        out_x[0:2] = x[0:2] / 30
        out_x[2:4] = x[3:5] / 30
        out_x[4:] = (x[6:] - 1000) / 3000
        return out_x, y

    init_dirname = r"F:\Week\20240414\Data\Model"
    batch_size = 8
    epochs = 200
    device = "cuda"
    model = YuYiFenGe().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=254).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    n_test = 2

    print("model:", model)
    print("init_dirname:", init_dirname)

    train_ds, test_ds = loadData()
    print("length train_ds:", len(train_ds))
    print("length test_ds:", len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print("length train_loader:", len(train_loader))
    print("length test_loader:", len(test_loader))

    def train():
        is_fit = True
        test_cm = ConfusionMatrix(5, ["beijing", "qiao", "guan", "cao", "beijing"])
        curr_dirname = timeDirName(init_dirname, True)
        cuur_dfn = DirFileName(curr_dirname)
        log_fn = cuur_dfn.fn("log.txt")
        writeCSVLine(log_fn, ["EPOCH", "BATCH", "LOSS", "OA", "FN"])
        copyFile(__file__, FN(__file__).changedirname(curr_dirname))
        print("curr_dirname:", curr_dirname)
        print("log_fn:", log_fn)

        def log():
            print("EPOCH: {0:>3d} BATCH: {1:>5d} LOSS: {2:>10.6f} OA: {3:>10.6f}".format(
                epoch, batchix, float(loss.cpu().item()), test_cm.OA()))
            fn = cuur_dfn.fn("model_{0}_{1}.pth".format(epoch, batchix))
            writeCSVLine(log_fn, [epoch, batchix, float(loss.cpu().item()), test_cm.OA(), fn])
            return fn

        def saveModel(to_fn):
            torch.save(model.state_dict(), to_fn)

        def tAcc():
            test_cm.clear()
            model.eval()
            with torch.no_grad():
                accs = []
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(device).float()
                    _logts = model(x)
                    y1 = torch.argmax(_logts, dim=1) + 1
                    y = y.numpy() + 1
                    y[y >= 5] = 5
                    y1[y1 >= 5] = 5
                    accs.append(np.sum(y == y1) / float(y.size))
                    # test_cm.addData(y.ravel(), y1.ravel())
            model.train()
            return np.mean(accs)

        def epochTAcc(is_save=False):
            if test_loader is not None:
                if not is_save:
                    if batchix % n_test == 0:
                        tAcc()
                        modname = log()
                        if is_save:
                            print("MODEL:", modname)
                            saveModel(modname)
                else:
                    tAcc()
                    modname = log()
                    print("MODEL:", modname)
                    saveModel(modname)

        if not is_fit:
            return

        for epoch in range(epochs):

            for batchix, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                x, y = x.float(), y.long()

                model.train()
                logts = model(x)
                loss = criterion(logts, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epochTAcc(False)

            epochTAcc(True)

            # scheduler.step(epoch + 1)

    train()
    pass


def main2():
    import  sys
    sys.path.append(r"F:\Week\20240414\Data\Desktop\Mobile_TransUnet")
    # F:\Week\20240414\Data\Desktop\Mobile_TransUnet\Transunet.py
    from Transunet import get_transNet
    net = get_transNet(4)
    print(net)

if __name__ == "__main__":
    r"""
    F:\PyCodes\Temp\JH_model.py
    python -c "import sys; sys.path.append(r'F:\PyCodes'); from Temp.JH_model import predict; predict()"
    """
    main2()
