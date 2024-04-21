# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHHModelImage.py
@Time    : 2024/3/30 10:27
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHHModelImage
-----------------------------------------------------------------------------"""
from SRTCodes.SRTModelImage import SRTModImPytorch


class SHHModImPytorch(SRTModImPytorch):

    def __init__(
            self,
            model_dir=None,
            model_name="PytorchModel",
            epochs=100,
            device="cuda",
            n_test=100,
            batch_size=32,
            n_class=2,
            class_names=None,
            win_size=()
    ):
        super().__init__(model_dir, model_name, epochs, device, n_test, batch_size, n_class, class_names, win_size)

    def imdcQD(self):
        self.imdc(
            to_geo_fn=None, geo_fns=None, grc=None, is_jdt=True, data_deal=None,
            is_print=True, description="Category"
        )


def main():
    pass


if __name__ == "__main__":
    main()
