# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : NumpyUtils.py
@Time    : 2023/7/4 20:13
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : BaseCodes of NumpyUtiles
-----------------------------------------------------------------------------"""
import os
import random

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from SRTCodes.Utils import Jdt, datasCaiFen


def saveCM(cm, save_cm_file, cate_names=None, infos=None):
    """ 保存混淆矩阵

    :param cm: 混淆矩阵
    :param save_cm_file: 混淆矩阵文件
    :param cate_names: 混淆矩阵类别名
    :param infos: 标识信息
    :return: 写到第几个了
    """
    if isinstance(infos, list):
        infos = " ".join(infos)
    if infos is None:
        infos = ""
    n_split = len(str(np.max(cm)))
    if n_split < 5:
        n_split = 5
    n_split += 1
    if cate_names is None:
        cate_names = ["C " + str(i) for i in range(cm.shpe[0])]
    for c in cate_names:
        if len(c) > n_split:
            n_split = len(c)
    if os.path.isfile(save_cm_file):
        with open(save_cm_file, "r", encoding="utf-8") as f:
            n = 0
            for line in f:
                if len(line) >= 2:
                    if line[:2] == "> ":
                        n += 1
            n += 1
    else:
        n = 1
    out_str = fmtCM(cm, cate_names)
    with open(save_cm_file, "a", encoding="utf-8") as f:
        f.write("> " + str(n) + " " + infos + "\n")
        f.write(out_str)
    return n


def fmtCM(cm: np.array, cate_names):
    fmt_row0 = "{:>8}"
    fmt_column0 = "{:>8}"
    fmt_number = "{:>8d}"
    fmt_float = "{:>8.2f}"
    n_cate = len(cate_names)
    out_s = ""
    out_s += fmt_column0.format("CM")
    for i in range(n_cate):
        out_s += " " + fmt_row0.format(cate_names[i])
    out_s += " " + fmt_row0.format("SUM")
    out_s += " " + fmt_row0.format("PA") + "\n"
    for i in range(n_cate):
        out_s += fmt_column0.format(cate_names[i])
        for j in range(n_cate):
            out_s += " " + fmt_number.format(int(cm[i, j]))
        out_s += " " + fmt_number.format(int(cm[i, n_cate]))
        out_s += " " + fmt_float.format(cm[i, n_cate + 1]) + "\n"
    out_s += fmt_column0.format("SUM")
    for i in range(n_cate):
        out_s += " " + fmt_number.format(int(cm[n_cate, i]))
    out_s += " " + fmt_number.format(int(cm[n_cate, n_cate]))
    out_s += " " + fmt_float.format(cm[n_cate, n_cate + 1]) + "\n"
    out_s += fmt_column0.format("UA")
    for i in range(n_cate):
        out_s += " " + fmt_float.format(cm[n_cate + 1, i])
    out_s += " " + fmt_float.format(cm[n_cate + 1, n_cate])
    out_s += " " + fmt_float.format(cm[n_cate + 1, n_cate + 1]) + "\n"
    return out_s


def printTable(d: np.array, columns_names=None, row_index=None, precision=2, alignment="right"):
    n_rows, n_colums = d.shape
    column_info = []
    # 获得行宽
    if columns_names is not None:
        for i in range(n_colums):
            column_info.append([len(columns_names[i]), precision])
    else:
        for i in range(n_colums):
            column_info.append([0, precision])
    d_max = np.max(d, axis=0)
    for i in range(n_colums):
        w = len(str(int(d_max[i]))) + precision
        if w > column_info[i][0]:
            column_info[i][0] = w + 2
    fmts = []
    for i in range(n_colums):
        fmt = ":"
        if alignment == "center":
            fmt += "^"
        elif alignment == "left":
            fmt += "<"
        else:
            fmt += ">"
        fmt += str(column_info[i][0]) + "." + str(precision) + "f"
        fmts.append("{" + fmt + "}")
    w = len(str(int(n_rows))) + 1
    fmt_index = "{:>" + str(w) + "d}"
    line0 = "|" + " " * w + " | "
    for j in range(n_colums):
        fmt = "{:"
        if alignment == "center":
            fmt += "^"
        elif alignment == "left":
            fmt += "<"
        else:
            fmt += ">"
        fmt += str(column_info[j][0])
        fmt += "}"
        if columns_names is not None:
            line0 += fmt.format(columns_names[j]) + " | "
        else:
            line0 += fmt.format(" ") + " | "

    line0 = line0[:-1]
    line1 = ""
    for c in line0:
        if c == "|":
            line1 += "+"
        else:
            line1 += "-"
    if columns_names is None:
        print(line1)
    else:
        print(line1)
        print(line0)
        print(line1)
    for i in range(n_rows):
        print("", end="|")
        print(fmt_index.format(i + 1), end=" | ")
        for j in range(n_colums):
            print(fmts[j].format(d[i, j]), end=" | ")
        print()
    print(line1)
    pass


def printDict(_dict: dict, fmts=None, alignment="right"):
    if fmts is None:
        fmts = ["{}" for _ in _dict]
    n_columns = len(_dict)
    n_rows = max([len(_dict[k]) for k in _dict])
    columns_names = list(_dict.keys())
    for k in _dict:
        _dict[k] = list(_dict[k])
        if len(_dict[k]) < n_rows:
            _dict[k].extend([""] * (n_rows - len(_dict[k])))
    column_info = []
    # 获得行宽
    for i in range(n_columns):
        column_info.append([len(columns_names[i]), precision])

    d_max = np.max(d, axis=0)
    for i in range(n_columns):
        w = len(str(int(d_max[i]))) + precision
        if w > column_info[i][0]:
            column_info[i][0] = w + 2

    fmts = []
    for i in range(n_columns):
        fmt = ":"
        if alignment == "center":
            fmt += "^"
        elif alignment == "left":
            fmt += "<"
        else:
            fmt += ">"
        fmt += str(column_info[i][0]) + "." + str(precision) + "f"
        fmts.append("{" + fmt + "}")

    w = len(str(int(n_rows))) + 1
    fmt_index = "{:>" + str(w) + "d}"
    line0 = "|" + " " * w + " | "
    for j in range(n_columns):
        fmt = "{:"
        if alignment == "center":
            fmt += "^"
        elif alignment == "left":
            fmt += "<"
        else:
            fmt += ">"
        fmt += str(column_info[j][0])
        fmt += "}"
        if columns_names is not None:
            line0 += fmt.format(columns_names[j]) + " | "
        else:
            line0 += fmt.format(" ") + " | "

    line0 = line0[:-1]
    line1 = ""
    for c in line0:
        if c == "|":
            line1 += "+"
        else:
            line1 += "-"
    if columns_names is None:
        print(line1)
    else:
        print(line1)
        print(line0)
        print(line1)
    for i in range(n_rows):
        print("", end="|")
        print(fmt_index.format(i + 1), end=" | ")
        for j in range(n_columns):
            print(fmts[j].format(d[i, j]), end=" | ")
        print()
    print(line1)


def neighborhood(d: np.ndarray, obj_func, rows: int, columns: int, dim=1):
    win_spl = [0, 0, 0, 0]
    win_spl[0] = 0 - int(rows / 2)
    win_spl[1] = 0 + round(rows / 2 + 0.1)
    win_spl[2] = 0 - int(columns / 2)
    win_spl[3] = 0 + round(columns / 2 + 0.1)
    row_start, row_end = int(rows / 2 + 0.1), d.shape[1] - int(rows / 2 + 0.1)
    column_start, column_end = int(columns / 2 + 0.1), d.shape[2] - int(columns / 2 + 0.1)

    d_ret = np.zeros([dim, d.shape[1], d.shape[2]])
    for i in range(row_start, row_end):
        for j in range(column_start, column_end):
            d_ret[:, i, j] = obj_func(d[:, i + win_spl[0]: i + win_spl[1], j + win_spl[2]: j + win_spl[3]])
    return d_ret


def reHist(d, ratio=0.001):
    n_re = d.shape[1] * d.shape[2] * ratio
    d0, d1 = [], []
    d00, d10 = 0, 0
    for i in range(d.shape[0]):
        d_i = d[i]
        print(i, ":", "-" * 80)
        while True:
            k1, k2 = 0, 0
            zuo, you = 0, 0
            h, bin_edges = np.histogram(d_i, bins=256)
            for j in range(h.shape[0]):
                zuo += h[j]
                k1 += 1
                d00 = bin_edges[j]
                if k1 == 10:
                    break
                if zuo >= n_re:
                    break
            for j in range(h.shape[0] - 1, -1, -1):
                you += h[j]
                k2 += 1
                d10 = bin_edges[j + 1]
                if k2 == 10:
                    break
                if you >= n_re:
                    break
            print(k1, d00, k2, d10)
            if k1 != 10 and k2 != 10:
                d0.append(d00)
                d1.append(d10)
                break
            d_i = np.clip(d_i, d00, d10)
    return np.array([d0, d1]).T


def reHist0(d_i, ratio=0.001, is_print=True):
    if is_print:
        print("-" * 80)
    n_re = int(np.size(d_i) * ratio)
    d00, d10 = 0, 0
    while True:
        k1, k2 = 0, 0
        zuo, you = 0, 0
        h, bin_edges = np.histogram(d_i, bins=256)
        for j in range(h.shape[0]):
            zuo += h[j]
            k1 += 1
            if k1 == 10:
                d00 = bin_edges[j]
                break
            if zuo >= n_re:
                d00 = bin_edges[j]
                break
        for j in range(h.shape[0] - 1, -1, -1):
            you += h[j]
            k2 += 1
            if k2 == 10:
                d10 = bin_edges[j + 1]
                break
            if you >= n_re:
                d10 = bin_edges[j + 1]
                break
        if is_print:
            print(k1, d00, k2, d10)
        if k1 != 10 and k2 != 10:
            d0, d1 = d00, d10
            break
        d_i = np.clip(d_i, d00, d10)
    return d0, d1


def npShuffle2(x: np.ndarray):
    select = [i for i in range(x.shape[0])]
    random.shuffle(select)
    d = []
    for i in select:
        d.append(x[i].tolist())
    return np.array(d)


def changePandasIndex(df: pd.DataFrame):
    df.index = [i for i in range(len(df))]
    return df


def filterEq(df, column_name, data):
    select = df[column_name].values
    select = select == data
    return df[select]


def minmaxData(d, d_min=None, d_max=None, to_min=0.0, to_max=1.0):
    if d_min is None:
        d_min = np.min(d)
    if d_max is None:
        d_max = np.max(d)
    d = np.clip(d, d_min, d_max)
    d = (d - d_min) / (d_max - d_min)
    d = d * (to_max - to_min) + to_min
    return d


def calPCA(d, num_components=None):
    d = d.T
    mean = np.mean(d, axis=0)
    centered_data = d - mean
    cov_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    if num_components is None:
        num_components = len(eigenvalues)
    selected_eigenvalues = eigenvalues[:num_components]
    selected_eigenvectors = eigenvectors[:, :num_components]
    transformed_data = np.dot(centered_data, selected_eigenvectors)
    return selected_eigenvalues, selected_eigenvectors, transformed_data


def calHSV(rgb):
    # 分离 RGB 通道
    r, g, b = rgb[0], rgb[1], rgb[2]

    # 计算最大值、最小值和差值
    max_val = np.max(rgb, axis=0)
    min_val = np.min(rgb, axis=0)
    delta = max_val - min_val + 0.000001

    # 初始化 HSV 数组
    h = np.zeros_like(max_val)
    s = np.zeros_like(max_val)
    v = max_val

    # Hue 计算
    mask = delta != 0
    h[mask & (max_val == r)] = (60 * (g - b) / delta)[mask & (max_val == r)] % 360
    h[mask & (max_val == g)] = (60 * (b - r) / delta + 120)[mask & (max_val == g)]
    h[mask & (max_val == b)] = (60 * (r - g) / delta + 240)[mask & (max_val == b)]

    # Saturation 计算
    s[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]

    # 将 HSV 合并为一个数组
    hsv = np.stack((h, s, v), axis=0)

    return hsv


def calHSI(rgb):
    # 分离 RGB 通道
    r, g, b = rgb[0], rgb[1], rgb[2]

    # 计算强度 I
    I = (r + g + b) / 3.0

    # 计算饱和度 S
    min_rgb = np.min(rgb, axis=0)
    S = 1 - 3 * min_rgb / (r + g + b + 1e-6)

    # 计算色调 H
    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    theta = np.arccos(np.clip(numerator / (denominator + 1e-6), -1, 1))

    H = np.zeros_like(theta)
    H[b > g] = 2 * np.pi - theta[b > g]
    H[b <= g] = theta[b <= g]

    H = H / (2 * np.pi)  # 将 H 归一化到 [0, 1] 范围

    # 将 H, S, I 合并为一个数组
    hsi = np.stack((H, S, I), axis=0)

    return hsi


def filterLee():
    # Define a Lee filter function
    def lee_filter(image, window_size):
        # Calculate the local mean using a convolution
        local_mean = uniform_filter(image, (window_size, window_size))

        # Calculate the local variance
        local_variance = uniform_filter(image ** 2, (window_size, window_size)) - local_mean ** 2

        # Estimate the noise variance
        noise_variance = local_variance.mean()

        # Calculate the filtered image
        filtered_image = local_mean + (image - local_mean) * np.minimum(
            noise_variance / (local_variance + noise_variance), 1)

        return filtered_image

    # Load your SAR image as a NumPy array
    # Replace 'your_image_array' with your SAR image array
    # Example:
    # your_image_array = np.array([[...]])

    # Set the window size for the filter (you can adjust this parameter)
    window_size = 3

    # # Apply the Lee filter to the SAR image
    # filtered_sar_image = lee_filter(your_image_array, window_size)
    #
    # # Display the original and filtered images
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(your_image_array, cmap='gray')
    # plt.title('Original SAR Image')
    #
    # plt.subplot(1, 2, 2)
    # plt.imshow(filtered_sar_image, cmap='gray')
    # plt.title('Filtered SAR Image (Lee Filter)')
    #
    # plt.show()


def scaleMinMax(d, d_min=None, d_max=None, is_01=True):
    if (d_min is None) and (d_max is None):
        is_01 = True
    if d_min is None:
        d_min = np.min(d)
    if d_max is None:
        d_max = np.max(d)
    d = np.clip(d, d_min, d_max)
    if is_01:
        d = (d - d_min) / (d_max - d_min)
    return d


def randomSampling(x, y, n=500, select_list=None):
    if select_list is None:
        select_list = [0, 1]
    random_list = [i for i in range(len(x))]
    random.shuffle(random_list)
    x, y = x[random_list], y[random_list]
    out_x, out_y = [], []
    for select_c in select_list:
        x1 = x[y == select_c]
        y1 = y[y == select_c]
        if len(y1) > n:
            x1 = x1[:n, :]
            y1 = y1[:n]
        out_x.append(x1)
        out_y.append(y1)
    out_x = np.concatenate(out_x)
    out_y = np.concatenate(out_y)
    random_list = [i for i in range(len(out_x))]
    random.shuffle(random_list)
    out_x, out_y = out_x[random_list], out_y[random_list]

    return out_x, out_y


def conv2dDim1(data, kernel, is_jdt=False):
    data, kernel = np.array(data), np.array(kernel)
    row, column = kernel.shape
    row_center, column_center = int(row / 2.0), int(column / 2.0)
    out_data = np.zeros_like(data)

    jdt = Jdt(len(list(range(row_center, data.shape[0] - row_center - 1))),
              "NumpyUtils::conv2dDim1").start(is_jdt=is_jdt)
    for i in range(row_center, data.shape[0] - row_center - 1):
        for j in range(column_center, data.shape[1] - column_center - 1):
            data_tmp = data[i - row_center:i + row_center + 1, j - column_center:j + column_center + 1]
            out_data[i, j] = np.sum(data_tmp * kernel)
        jdt.add()
    jdt.end()

    return out_data


class NumpyDataCenter:

    def __init__(self, dim=2, win_size=None, spl_size=None):
        self.dim = dim
        self.spl_size = spl_size
        self.win_size = win_size
        self.win_spl = [0, 0, 0, 0]
        self.row = 0
        self.column = 0
        self.range = [0, 0, 0, 0]

        self.initWinSize(win_size)
        self.initSampleSize(spl_size)
        self.initRange()

    def copy(self):
        return NumpyDataCenter(self.dim, self.win_size, self.spl_size)

    def initWinSize(self, win_size=None):
        self.win_size = win_size
        if self.win_size is None:
            return
        row_size, column_size = win_size[0], win_size[1]
        self.win_spl[0] = 0 - int(row_size / 2)
        self.win_spl[1] = 0 + round(row_size / 2 + 0.1)
        self.win_spl[2] = 0 - int(column_size / 2)
        self.win_spl[3] = 0 + round(column_size / 2 + 0.1)

    def initSampleSize(self, spl_size=None):
        self.spl_size = spl_size
        if self.spl_size is None:
            return
        self.row = int(self.spl_size[0] / 2.0)
        self.column = int(self.spl_size[1] / 2.0)

    def initRange(self):

        self.range[0] = self.row + self.win_spl[0]
        self.range[1] = self.row + self.win_spl[1]
        self.range[2] = self.column + self.win_spl[2]
        self.range[3] = self.column + self.win_spl[3]

    def fit(self, x):
        if self.win_size is None:
            return x
        if self.spl_size is None:
            if self.dim == 2:
                self.spl_size = x.shape
            elif self.dim == 3:
                self.spl_size = x.shape[1:]
            elif self.dim == 4:
                self.spl_size = x.shape[2:]
            self.initSampleSize(self.spl_size)
            self.initRange()

        if self.dim == 2:
            out_x = x[self.range[0]:self.range[1], self.range[2]:self.range[3]]
        elif self.dim == 3:
            out_x = x[:, self.range[0]:self.range[1], self.range[2]:self.range[3]]
        elif self.dim == 4:
            out_x = x[:, :, self.range[0]:self.range[1], self.range[2]:self.range[3]]
        else:
            return x

        return out_x

    def fit2(self, x):
        if self.win_size is None:
            return x
        if self.spl_size is None:
            if self.dim == 2:
                self.spl_size = x.shape
            elif self.dim == 3:
                self.spl_size = x.shape[1:]
            self.initSampleSize(self.spl_size)
            self.initRange()

        out_x = x[:, :, self.range[0]:self.range[1], self.range[2]:self.range[3]]
        return out_x


def dataCenter(data):
    if len(data.shape) == 2:
        # 0 1 2 3 4 5 6
        return data[int(data.shape[0] / 2.0), int(data.shape[1] / 2.0)]
    elif len(data.shape) == 3:
        return data[:, int(data.shape[1] / 2.0), int(data.shape[2] / 2.0)]
    else:
        return None


def connectedComponent(image, is_jdt=False, is_ret_xys=False):
    labels = {}
    collection = {}
    current_label = 1
    jdt = Jdt(image.shape[0], "connectedComponent").start(is_jdt=is_jdt)
    for y in range(image.shape[0]):
        jdt.add(is_jdt=is_jdt)
        for x in range(image.shape[1]):
            if image[y, x] != 0:
                neighbors = []
                if y > 0 and image[y - 1, x] != 0:
                    neighbors.append(labels[(y - 1, x)])
                if x > 0 and image[y, x - 1] != 0:
                    neighbors.append(labels[(y, x - 1)])

                if len(neighbors) == 0:
                    labels[(y, x)] = current_label
                    collection[current_label] = [(y, x)]
                    current_label += 1
                else:
                    min_neighbor = min(neighbors)
                    labels[(y, x)] = min_neighbor
                    collection[min_neighbor].append((y, x))
                    is_tiao = False
                    for neighbor in neighbors:
                        if neighbor != min_neighbor:
                            collection[min_neighbor].extend(collection.pop(neighbor))
                            is_tiao = True
                    if is_tiao:
                        for d in collection[min_neighbor]:
                            labels[d] = min_neighbor
    jdt.end(is_jdt=is_jdt)

    current_label = 1
    out_image = np.zeros_like(image)
    for d in collection.values():
        for i, j in d:
            out_image[i, j] = current_label
        current_label += 1

    if is_ret_xys:
        return out_image, collection
    return out_image


def categoryMap(categorys, map_dict, is_notfind_to0=False):
    if isinstance(map_dict, list):
        map_dict = {i: map_dict[i] for i in range(len(map_dict))}
    to_category = []
    for d in categorys:
        if d in map_dict:
            to_category.append(map_dict[d])
        else:
            if is_notfind_to0:
                to_category.append(0)
            else:
                to_category.append(d)
    return to_category


def npyShape(npy_fn):
    with open(npy_fn, 'rb') as fp:
        from numpy.lib.format import _read_array_header
        shape, fortran_order, dtype = _read_array_header(fp, version=(1, 0), max_header_size=10000)

    return shape


class _TSN:

    def __init__(self, *names, dim=0):
        self.names = list(datasCaiFen(names))
        self.select_names = None
        self.numbers = []
        self.dim = dim

    def selectNames(self, *names):
        names = list(datasCaiFen(names))
        self.select_names = names
        self.numbers = []
        for name in names:
            self.numbers.append(self.names.index(name))
        return self

    def fit(self, data):
        if self.select_names is None:
            return data
        if self.dim == 0:
            return data[self.numbers]
        elif self.dim == 1:
            return data[:, self.numbers]

    def length(self):
        return len(self.select_names)

    def toDict(self):
        return {"names": self.names, "select_names": self.select_names, "numbers": self.numbers, "dim": self.dim}


class TensorSelectNames:

    def __init__(self, *names, dim=0):
        self.names = list(datasCaiFen(names))
        self.dim = dim
        self.tsns = {}

    def __getitem__(self, item) -> _TSN:
        return self.tsns[item]

    def __len__(self):
        return len(self.tsns)

    def addTSN(self, tsn_name, select_names, names=None, dim=None):
        if names is None:
            names = self.names
        if dim is None:
            dim = self.dim
        self.tsns[tsn_name] = _TSN(*names, dim=dim).selectNames(select_names)

    def toDict(self):
        return {"names": self.names, "dim": self.dim, "tsns": {tsn: self.tsns[tsn].toDict() for tsn in self.tsns}}

    def length(self, *names):
        if len(names) == 0:
            names = list(self.tsns.keys())
        return sum(list(self.tsns[name].length() for name in names))


def eig(a, b, c, d, is_01=False):
    k1 = np.sqrt(0.25 * a ** 2 - 0.5 * a * d + b ** 2 + c ** 2 + 0.25 * d ** 2)
    k2 = (a + d) / 2.0
    e1 = k2 + k1
    e2 = k2 - k1
    k3 = b - c * 1j
    v11 = (e1 - d) / k3
    v21 = (e2 - d) / k3
    if is_01:
        v11 = v11 / np.sqrt(1 + v11 ** 2)
        v21 = v21 / np.sqrt(1 + v21 ** 2)
    return e1, e2, v11, v21


def eig2(a11, a12, a21, a22):
    k1 = np.sqrt((a11 - a22) ** 2 + 4 * a12 * a21)
    e1 = ((a11 + a22) - k1) / 2
    e2 = ((a11 + a22) + k1) / 2
    d1 = np.abs(a21) ** 2
    d2 = np.abs((e1 - a11)) ** 2
    norm1 = np.sqrt(d1 + d2)
    d2 = np.abs((e2 - a11)) ** 2
    norm2 = np.sqrt(d1 + d2)
    v11 = a12 / norm1
    v21 = (e1 - a11) / norm1
    v12 = a12 / norm2
    v22 = (e2 - a11) / norm2
    return e1, e2, v11, v12, v21, v22


def update10EDivide10(_data):
    return np.power(10.0, _data / 10.0)


def update10Log10(_data):
    return 10 * np.log10(_data + 0.0001)


class NumpySampling:

    def __init__(self, win_row, win_column):
        self.win_spl = [0, 0, 0, 0]
        self.win_spl[0] = 0 - int(win_row / 2)
        self.win_spl[1] = 0 + round(win_row / 2 + 0.1)
        self.win_spl[2] = 0 - int(win_column / 2)
        self.win_spl[3] = 0 + round(win_column / 2 + 0.1)
        self.data = None

    def get(self, row, column):
        return self.data[:, row + self.win_spl[0]: row + self.win_spl[1],
               column + self.win_spl[2]: column + self.win_spl[3]]


def main():
    return


if __name__ == "__main__":
    main()
