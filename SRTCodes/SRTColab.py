# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTColab.py
@Time    : 2024/8/11 20:25
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of HZColab
-----------------------------------------------------------------------------"""
import torch

from SRTCodes.GDALTorch import GDALTorchImdc
from SRTCodes.Utils import Jdt


def torchImdc(model, raster_fn, to_imdc_fn, x_keys, data_deal, win_size, color_table, read_size, is_save_tiles,
              func_predict=None, device="cuda", fun_print=print):
    if func_predict is None:
        def func_logit_category(x: torch.Tensor):
            with torch.no_grad():
                logit = model(x)
                y = torch.argmax(logit, dim=1) + 1
            return y

        func_predict = func_logit_category

    model.eval()
    model.zero_grad()

    gti = GDALTorchImdc(raster_fn)
    gti.imdc3(
        func_predict=func_predict, win_size=win_size, to_imdc_fn=to_imdc_fn,
        fit_names=x_keys, data_deal=data_deal, color_table=color_table,
        is_jdt=True, device=device, read_size=read_size, is_save_tiles=is_save_tiles,
        fun_print=fun_print,
    )

    model.train()

    Jdt


def main():
    pass


if __name__ == "__main__":
    main()
