# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2Training.py
@Time    : 2024/7/9 16:41
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2Training
-----------------------------------------------------------------------------"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from SRTCodes.GDALTorch import GDALTorchImdc
from SRTCodes.GDALUtils import GDALRasterClip
from SRTCodes.PytorchModelTraining import TorchTraining
from SRTCodes.Utils import changext
from Shadow.Hierarchical.SHH2Config import GET_RASTER_FN
from Shadow.Hierarchical.SHH2Model import SamplesData


def getCSVFn(city_name):
    if city_name == "qd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\25\vhl\sh2_spl25_vhl_2_spl2.csv"
    elif city_name == "cd":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\30\cd\sh2_spl30_cd6_spl.csv"
    elif city_name == "bj":
        csv_fn = r"F:\ProjectSet\Shadow\Hierarchical\Samples\27\3\sh2_spl273_5_spl.csv"
    else:
        raise Exception("City name \"{}\"".format(city_name))
    return csv_fn


def x_deal(_x):
    for i in range(0, 6):
        _x[i] = _x[i] / 3000
    for i in range(6, 10):
        _x[i] = (_x[i] + 30) / 35
    for i in range(12, 16):
        _x[i] = (_x[i] + 30) / 35
    return _x


def _noneFunc(_data, _none_data):
    if _data is None:
        _data = _none_data
    return _data


def loadDS(city_name, win_size=None, read_size=None, map_dict=None, get_names=None):
    win_size = _noneFunc(win_size, (21, 21))
    read_size = _noneFunc(read_size, (21, 21))
    map_dict = _noneFunc(map_dict, {
        "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
    })
    get_names = _noneFunc(get_names, [
        #  0  1  2  3  4  5
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
        #  6  7  8  9 10 11
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        # 12 13 14 15 16 17
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ])

    csv_fn = getCSVFn(city_name)
    sd = SamplesData()
    sd.addDLCSV(csv_fn, (21, 21), get_names, x_deal)
    samples = sd.dltorch(
        map_dict, win_size, read_size,
        train_filters=[("TEST", "==", 1)], test_filter=[("TEST", "==", 0)],
        device="cuda:0",
    )

    return samples.train_ds, samples.test_ds


def loadLoader(city_name, win_size=None, read_size=None, map_dict=None, get_names=None, batch_size=32):
    train_ds, test_ds = loadDS(
        city_name=city_name, win_size=win_size,
        read_size=read_size, map_dict=map_dict, get_names=get_names)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def training(
        model,
        train_ds,
        test_ds,
        epochs=100,
        n_test=20,
        batch_size=32,
        save_model_fmt=None,
        n_epoch_save=50,
        func_epoch=None,
        func_xy_deal=None,
        func_batch=None,
        func_loss_deal=None,
        func_y_deal=lambda y: y + 1,
        func_logit_category=None,
        func_print=print,
        func_field_record_save=None,
):
    torch_training = TorchTraining()
    torch_training.model = model
    torch_training.criterion = nn.CrossEntropyLoss()
    torch_training.epochs = epochs
    torch_training.device = "cuda:0"
    torch_training.n_test = n_test

    torch_training.trainLoader(train_ds, batch_size=batch_size)
    torch_training.testLoader(test_ds, batch_size=batch_size)

    torch_training.optimizer(optim.Adam, lr=0.001, eps=0.000001)
    torch_training.scheduler(optim.lr_scheduler.StepLR, step_size=20, gamma=0.6, last_epoch=-1)

    torch_training.initCM(3)
    torch_training.save_model_fmt = save_model_fmt
    torch_training.n_epoch_save = n_epoch_save

    if func_logit_category is None:
        def func_logit_category(_model, x: torch.Tensor):
            logit = _model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

    torch_training.func_epoch = func_epoch
    torch_training.func_xy_deal = func_xy_deal
    torch_training.func_batch = func_batch
    torch_training.func_loss_deal = func_loss_deal
    torch_training.func_y_deal = func_y_deal

    torch_training.func_logit_category = func_logit_category
    torch_training.func_print = func_print
    torch_training.func_field_record_save = func_field_record_save

    torch_training.train()


def imdcing(city_name, model, win_size, mod_fn=None, to_imdc_fn=None,
            func_logit_category=None, x=None, y=None,
            rows=301, columns=301):
    if mod_fn is not None:
        model.load_state_dict(torch.load(mod_fn))
        model.to("cuda:0")
    model.eval()

    if func_logit_category is None:
        def func_logit_category(_model, x: torch.Tensor):
            logit = _model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

    def func_predict(x):
        return func_logit_category(model, x)

    if to_imdc_fn is None:
        if mod_fn is not None:
            to_imdc_fn = changext(mod_fn, "_imdc.tif")

    if to_imdc_fn is None:
        to_imdc_fn = r"F:\ProjectSet\Shadow\Hierarchical\Images\Temp\_SHH2Training_image_imdc.tif"

    raster_fn = GET_RASTER_FN(city_name)
    if (x is None) or (y is None):
        if city_name == "qd":
            x, y = 120.448623, 36.131374
        elif city_name == "bj":
            x, y = 116.577230, 39.728331
        elif city_name == "cd":
            x, y = 104.066829, 30.782020
        else:
            raise Exception("Can not init x, y for city name \"{}\"".format(raster_fn))

    gr = GDALRasterClip(raster_fn)
    fn = gr.coorCenter(
        r"F:\ProjectSet\Shadow\Hierarchical\Images\Temp\_SHH2Training_image.tif",
        x=x, y=y, rows=rows, columns=columns,
    )

    x_keys = [
        #  0  1  2  3  4  5
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
        #  6  7  8  9 10 11
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        # 12 13 14 15 16 17
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ]
    gti = GDALTorchImdc(fn)
    gti.imdc2(
        func_predict=func_predict, win_size=win_size, to_imdc_fn=to_imdc_fn,
        fit_names=x_keys, data_deal=x_deal, color_table={
            1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), },
        n=-1, is_jdt=True, device="cuda:0",
    )

    model.train()
    return to_imdc_fn


def main():
    city_name = "qd"

    win_size = 21, 21
    read_size = 21, 21

    get_names = [
        #  0  1  2  3  4  5
        "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
        #  6  7  8  9 10 11
        "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
        # 12 13 14 15 16 17
        "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
    ]
    map_dict = {
        "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
        "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
    }

    samples = loadDS(city_name, win_size, read_size, map_dict, get_names)

    print(samples)


if __name__ == "__main__":
    main()
