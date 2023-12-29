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


def main():
    return


if __name__ == "__main__":
    main()


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
