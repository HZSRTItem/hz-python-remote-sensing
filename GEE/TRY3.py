import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from SRTCodes.Utils import Jdt


def readTFData(tfrecord_file):
    dirname = os.path.split(tfrecord_file)[0]
    fn = os.path.splitext(os.path.split(tfrecord_file)[1])[0]
    data_filename = os.path.join(dirname, fn + "_data.npy")
    label_filename = os.path.join(dirname, fn + "_label.npy")
    property_filename = os.path.join(dirname, fn + "_pro.json")
    crf_filename = os.path.join(dirname, fn + "_crf.npy")
    image_data = np.load(data_filename)
    label_data = np.load(label_filename)
    with open(property_filename, "r", encoding="utf-8") as f:
        property_data = json.load(f)

    return image_data, label_data, property_data, crf_filename


def filterFileExt(dirname=".", ext=""):
    filelist = []
    for fn in os.listdir(dirname):
        fn = os.path.join(dirname, fn)
        if os.path.isfile(fn):
            if os.path.splitext(fn)[1] == ext:
                filelist.append(fn)
    return filelist


def readJson(json_fn):
    with open(json_fn, "r", encoding="utf-8") as f:
        property_data = json.load(f)
    return property_data


def geojsonsToNpys(dirname, data_fields=None, save_fields=None):
    fns = filterFileExt(dirname, ".geojson")


def geojsonsToNpy(dirname, data_fields=None, save_fields=None):
    if save_fields is None:
        save_fields = []
    if data_fields is None:
        data_fields = []
    fns = filterFileExt(dirname, ".geojson")
    data_list = []
    feat_list = []
    to_fn = fns[0]
    to_fn_list = to_fn.split("_")[:-1]
    to_fn = "_".join(to_fn_list)
    to_fn = os.path.join(dirname, to_fn)
    to_csv_fn = to_fn + "_fields.csv"
    to_data_fn = to_fn + "_data.npy"
    print(to_csv_fn, to_data_fn)
    jdt = Jdt(len(fns), "Geojsons To Npy")
    jdt.start()
    for fn in fns:
        json_dict = readJson(fn)
        for feat in json_dict["features"]:
            data_list.append([feat["properties"][field] for field in data_fields])
            feat_list.append({field: feat["properties"][field] for field in save_fields})
            feat_list[-1]["X"] = feat["geometry"]["coordinates"][0]
            feat_list[-1]["Y"] = feat["geometry"]["coordinates"][1]
        jdt.add()
    jdt.end()
    df = pd.DataFrame(feat_list)
    df.to_csv(to_csv_fn)
    data = np.array(data_list)
    np.save(to_data_fn, data.astype("int16"))
    print(df)
    print(data.shape)


class LMDataSet(Dataset):

    def __init__(self, np_filename, json_filename):
        super(LMDataSet, self).__init__()

        self.np_fn = np_filename
        self.json_fn = json_filename
        self.data = None
        self.labels = None

        self.readData()

    def readData(self):
        self.data = np.load(self.np_fn)
        json_data = readJson(self.json_fn)
        self.labels = [json_d["Name"] for json_d in json_data]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        x = x / 1600
        y = self.labels[index]
        return x, y


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=0, bias=False)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=0, bias=False)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(8, 4, kernel_size=3, padding=0, bias=False)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(4, 2, kernel_size=7, padding=0, bias=False)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        # 第一阶段
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        # x = self.relu4(x)

        x = torch.squeeze(x)
        return x


init_dir = r"D:\LM_cov\Mods"
current_time = datetime.now()
formatted_time = current_time.strftime("%Y%m%d%H%M%S")
to_dir = os.path.join(init_dir, formatted_time)
# if not os.path.isdir(to_dir):
#     os.mkdir(to_dir)
acc_text_filename = os.path.join(to_dir, "{0}_acc.csv".format(formatted_time))
code_text_filename = os.path.join(to_dir, "{0}_code.py".format(formatted_time))


# with open(code_text_filename, "w", encoding="utf-8") as f:
#     with open(__file__, "r", encoding="utf-8") as fr:
#         text = fr.read()
#     f.write(text)


def writeText(filename, line: list):
    with open(filename, "a", encoding="utf-8") as f:
        if len(line) == 1:
            f.write(str(line[0]))
        else:
            f.write(str(line[0]))
            for i in range(1, len(line)):
                f.write(",")
                f.write(str(line[i]))
        f.write("\n")


def getModelFileName(epoch):
    return os.path.join(to_dir, "{0}_{1}.pth".format(formatted_time, epoch))


def train():
    epochs = 100
    batch_size = 128

    train_dataset = LMDataSet(
        np_filename=r"D:\LM_cov\LM_tf_13\lm_sample_train\lm_crf_data.npy",
        json_filename=r"D:\LM_cov\LM_tf_13\lm_sample_train\lm_crf_pro.json",
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = LMDataSet(
        np_filename=r"D:\LM_cov\LM_tf_13\lm_sample_test\lm_sample_test_data.npy",
        json_filename=r"D:\LM_cov\LM_tf_13\lm_sample_test\lm_sample_test_pro.json",
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    n_test = 10

    device = "cpu"

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 学习率

    writeText(acc_text_filename, ["Epoch", "Accuracy", "Loss"])
    for epoch in range(epochs):
        loss = None

        for batchix, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x, y = x.float(), y.long()

            logts = model(x)  # 模型训练
            loss = criterion(logts, y)  # 损失函数

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化迭代

            # 测试 ------------------------------------------------------------------
            if test_loader is not None:
                if batchix % n_test == 0:
                    acc = tAcc(model, test_loader)
                    print("Epoch:{0:2d} Batch:{1:3d} Loss:{3:10.6f} Acc:{2:6.2f}%".format(
                        epoch, batchix, acc, loss.item()))

        acc = tAcc(model, test_loader)
        print("Epoch:{0:2d} Acc:{1:6.2f}% Loss:{2:10.6f} ".format(epoch, acc, loss.item()))
        writeText(acc_text_filename, [epoch, acc, loss.item()])
        print("-" * 60)
        # 测试
        mod_fn = getModelFileName(epoch)  # 设计一个明
        torch.save(model.state_dict(), mod_fn)


def tAcc(model, test_loader):
    with torch.no_grad():
        total_number = 0
        n_true = 0
        for i, (x, y) in enumerate(test_loader):
            logts = model(x)
            y_pred = torch.argmax(logts, dim=1)
            total_number += len(y_pred)
            n_true += torch.sum(y == y_pred).item()
            pass
        acc = n_true * 1.0 / total_number
    return acc * 100


def toGEE():
    mod_filename = r"D:\LM_cov\Mods\20240116220558\20240116220558_66.pth"
    mod = Net()
    mod.load_state_dict(torch.load(mod_filename))
    to_dict = mod.state_dict()

    for k in to_dict:
        key_name = k.split(".")[0]
        # print(key_name)
        data = to_dict[k]
        for i in range(data.shape[0]):
            data_i = data[i]
            # print("{0}: SHAPE {1}".format(i, data_i.shape), end="\n")
            for j in range(data.shape[1]):
                var_name = "{0}_{1}_{2}".format(key_name, i, j)
                print("var {0} = ee.Kernel.fixed({1}, {2}, {3});".format(
                    var_name, data.shape[2], data.shape[3], data_i[j].tolist()), end="\n")

    for k in to_dict:
        key_name = k.split(".")[0]
        data = to_dict[k]
        name1 = "{0}_ker_list".format(key_name)
        print("var {0} = [".format(name1))
        for i in range(data.shape[0]):
            print("[", end="")
            for j in range(data.shape[1]):
                var_name = "{0}_{1}_{2}".format(key_name, i, j)
                print("{0}, ".format(var_name), end="")
            print("],")
        print("];")

    init_name = "planet_im"
    for k in to_dict:
        key_name = k.split(".")[0]
        data = to_dict[k]
        name1 = "{0}_ker_list".format(key_name)
        name2 = "{0}_out_im".format(key_name)
        print("var {0} = convNtoN2({1}, {2}, {3}, {4}, \"{5}\");".format(
            name2, init_name, name1, data.shape[1], data.shape[0], name2))
        print("{0} = {0}.gt(0).multiply({0});".format(name2))
        init_name = name2
        print("print(\"{0}\", {0});".format(name2))

    # for k in to_dict:
    #     print("*"*10, k, "*"*10,)
    #     key_name = k.split(".")[0]
    #     data = to_dict[k]
    #     for i in range(data.shape[0]):
    #         data_i = data[i]
    #         print("> {0}: SHAPE {1}".format(i, data_i.shape), end="\n")
    #         for j in range(data.shape[1]):
    #             var_name = "{0}_{1}_{2}".format(key_name, i, j)
    #             print("  + {0}: {1}".format(j, var_name))
    #
    # print("MODEL: ")
    # print(mod)

    pass


def main():
    # toGEE()
    geojsonsToNpy(r"F:\ProjectSet\FiveMeter\Samples\2\drive-download-20240117T075435Z-001",["B", "G", "R", "N"], ["SRT", "category"])
    pass


if __name__ == "__main__":
    main()
