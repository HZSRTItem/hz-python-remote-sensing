# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ToLinNet.py
@Time    : 2024/1/18 21:26
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of ToLinNet
-----------------------------------------------------------------------------"""

import json
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


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


# ----------------------------------------------------- Model ----------------------------------------------------------

def tensorCenter(tensor: torch.Tensor, size):
    row, column = int(tensor.shape[2] / 2.0), int(tensor.shape[3] / 2.0)
    size_row, size_column = int(size[0] / 2.0), int(size[1] / 2.0)
    tensor_out = tensor[:, :, row - size_row:row + size_row + 1, column - size_column:column + size_column + 1]
    return tensor_out


class NetBranch(nn.Sequential):

    def __init__(self, in_channels, channels=None, kernel_sizes=None, biases=None):
        super(NetBranch, self).__init__()
        if channels is None:
            channels = []
        if kernel_sizes is None:
            kernel_sizes = [3 for _ in range(len(channels))]
        if biases is None:
            biases = [False for _ in range(len(channels))]

        layers = []

        for i, channel in enumerate(channels):
            layers.append(nn.Conv2d(
                in_channels=in_channels, out_channels=channel, kernel_size=kernel_sizes[i], bias=biases[i]
            ))
            layers.append(nn.ReLU())
            in_channels = channel

        self.layers = nn.Sequential(*tuple(layers))

    def forward(self, x):
        # for i, layer in enumerate(self.layers):
        #     x = layer(x)
        x = self.layers(x)
        return x


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        """
         1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 
         1  3  5  7  9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 
        """

        self.branch7 = NetBranch(in_channels=4, channels=[16, 32, 8], kernel_sizes=[3, 3, 3])  # [7, 7]
        self.branch13 = NetBranch(in_channels=4, channels=[8, 16, 16, 8, 8], kernel_sizes=[3, 3, 3, 3, 5])  # [13, 13]
        self.branch25 = NetBranch(in_channels=4, channels=[8, 16, 16, 8, 8, 8], kernel_sizes=[5, 5, 5, 5, 5, 5])  # 25
        self.branch47 = NetBranch(in_channels=4, channels=[8, 16, 16, 8, 8, 8], kernel_sizes=[9, 9, 9, 9, 9, 7])  # 47

        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        # 第一阶段
        x7 = tensorCenter(x, (7, 7))
        x7 = self.branch7(x7)
        x13 = tensorCenter(x, (13, 13))
        x13 = self.branch13(x13)
        x25 = tensorCenter(x, (25, 25))
        x25 = self.branch25(x25)
        x47 = tensorCenter(x, (47, 47))
        x47 = self.branch47(x47)

        x = x7 + x13 + x25 + x47
        x = torch.squeeze(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------


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


def toGEE2():
    mod_filename = r"F:\Week\20240121\Data\mod2.pth"
    mod = Net()
    mod.load_state_dict(torch.load(mod_filename))
    to_dict = mod.state_dict()
    print(mod)

    conv_branchs = {}
    for k, v in to_dict.items():
        if "branch" not in k:
            continue
        key_name = k.split(".")[0]
        if key_name not in conv_branchs:
            conv_branchs[key_name] = []
        conv_branchs[key_name].append(v)

    print(conv_branchs.keys())

    def format_branch(branch_name):
        f_togee = open(mod_filename + "_{0}_togee.txt".format(branch_name), "w", encoding="utf-8")

        for i, data in enumerate(conv_branchs[branch_name]):
            for i in range(data.shape[0]):
                data_i = data[i]
                for j in range(data.shape[1]):
                    var_name = "{0}_{1}_{2}".format(branch_name, i, j)
                    print("var {0} = ee.Kernel.fixed({1}, {2}, {3});".format(
                        var_name, data.shape[2], data.shape[3], data_i[j].tolist()), end="\n", file=f_togee)

        for data in conv_branchs[branch_name]:
            key_name = branch_name
            name1 = "{0}_ker_list".format(key_name)
            print("var {0} = [".format(name1), file=f_togee)
            for i in range(data.shape[0]):
                print("[", end="", file=f_togee)
                for j in range(data.shape[1]):
                    var_name = "{0}_{1}_{2}".format(key_name, i, j)
                    print("{0}, ".format(var_name), end="", file=f_togee)
                print("],", file=f_togee)
            print("];", file=f_togee)

        init_name = "planet_im"

        for i, data in enumerate(conv_branchs[branch_name]):
            key_name = branch_name
            name1 = "{0}_ker_list".format(key_name)
            name2 = "{0}_{1}_out_im".format(key_name, i)
            print("var {0} = convNtoN2({1}, {2}, {3}, {4}, \"{5}\");".format(
                name2, init_name, name1, data.shape[1], data.shape[0], name2), file=f_togee)
            print("{0} = {0}.gt(0).multiply({0});".format(name2), file=f_togee)
            init_name = name2
            print("print(\"{0}\", {0});".format(name2), file=f_togee)

        f_togee.close()

    format_branch("branch13")


def main():
    # toGEE()
    # mod = Net()
    # x = torch.rand((32, 4, 49, 49))
    # x = mod(x)
    # torch.save(mod.state_dict(), r"F:\Week\20240121\Data\mod2.pth")
    # mod.load_state_dict(torch.load(r"F:\Week\20240121\Data\mod2.pth"))
    # print(mod)
    # to_dict = mod.state_dict()
    toGEE2()

    pass


if __name__ == "__main__":
    main()
