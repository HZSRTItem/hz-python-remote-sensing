# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SHH2DL2.py
@Time    : 2024/7/7 20:17
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SHH2DL2
-----------------------------------------------------------------------------"""
from functools import partial
from typing import Callable, Optional, Any, List, Sequence, Type, Union

import torch
from torch import nn
from torchvision.models import googlenet, vgg11
from torchvision.models.convnext import CNBlockConfig, CNBlock, LayerNorm2d
from torchvision.models.resnet import BasicBlock, conv1x1
from torchvision.models.video.resnet import Bottleneck
from torchvision.ops import Conv2dNormActivation

from SRTCodes.NumpyUtils import NumpyDataCenter
from SRTCodes.SRTTimeDirectory import TimeDirectory
from SRTCodes.Utils import filterFileEndWith, DirFileName, FieldRecords, SRTWriteText
from Shadow.Hierarchical import SHH2Config
from Shadow.Hierarchical.SHH2DLModels import SHH2MOD_SpectralTextureDouble
from Shadow.Hierarchical.SHH2Model import SamplesData, TorchModel


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


def getRasterFileNames(city_name):
    if city_name == "qd":
        raster_fns = filterFileEndWith(r"F:\ProjectSet\Shadow\Hierarchical\Images\QingDao\SH22\Tiles", end=".tif")
    elif city_name == "cd":
        raster_fns = filterFileEndWith(r"F:\ProjectSet\Shadow\Hierarchical\Images\ChengDu\SH22\Tiles", end=".tif")
    elif city_name == "bj":
        raster_fns = filterFileEndWith(r"F:\ProjectSet\Shadow\Hierarchical\Images\BeiJing\SH22\Tiles", end=".tif")
    else:
        raise Exception("City name \"{}\"".format(city_name))
    return raster_fns


def x_deal(_x):
    for i in range(0, 6):
        _x[i] = _x[i] / 3000
    for i in range(6, 10):
        _x[i] = (_x[i] + 30) / 35
    for i in range(12, 16):
        _x[i] = (_x[i] + 30) / 35
    return _x


def getModel1(get_names):
    ndc3 = NumpyDataCenter(4, (3, 3), (21, 21))

    """
        tsn.addTSN("OPT", ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2", ])
        tsn.addTSN("BS", ["AS_VV", "DE_VV", "AS_VH", "DE_VH"])
        tsn.addTSN("C2", ["AS_C11", "DE_C11", "AS_C22", "DE_C22"])
        tsn.addTSN("HA", ["AS_H", "DE_H", "AS_Alpha", "DE_Alpha"])
    """

    def to_3d(x):
        x1 = torch.zeros(x.shape[0], 2, 2, x.shape[2], x.shape[3]).to(x.device)
        x1[:, 0, :, :, :] = x[:, [0, 1], :, :]
        x1[:, 1, :, :, :] = x[:, [2, 3], :, :]
        return x1

    def xforward(x: torch.Tensor):
        x_opt = x[:, [0, 1, 2, 3, 4, 5]]
        x_bs = x[:, [6, 12, 7, 13]]
        x_c2 = x[:, [8, 14, 9, 15]]
        x_ha = x[:, [10, 16, 11, 17]]
        x0 = ndc3.fit(torch.cat([x_opt, x_bs, x_c2, x_ha], dim=1))
        return x0, x_opt[:, [2, 3], :, :], to_3d(x_bs), to_3d(x_c2), to_3d(x_ha)

    model = SHH2MOD_SpectralTextureDouble(len(get_names), 4, blocks_type="Transformer", is_texture=True)
    model.xforward = xforward
    return model


def getModel2(get_names):
    model = nn.Sequential(
        nn.Conv2d(len(get_names), len(get_names), 3, 1, 1),
        nn.Flatten(start_dim=1),
        nn.Linear(21 * 21 * len(get_names), 4),
    )
    return model


class ConvNeXt(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            in_ch=3,
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                in_ch,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            in_ch=3,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_ch, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def getTorchInitModel(model_name, n_categorys, in_ch):
    model = None
    if model_name == "AlexNet":
        model = AlexNet(num_classes=n_categorys, in_ch=in_ch)
    elif model_name == "ConvNeXt":
        block_setting = [
            CNBlockConfig(96, 192, 3),
            CNBlockConfig(192, 384, 3),
            CNBlockConfig(384, 768, 9),
            CNBlockConfig(768, None, 3),
        ]
        model = ConvNeXt(num_classes=n_categorys, in_ch=in_ch, block_setting=block_setting)
    elif model_name == "GoogLeNet":
        model = googlenet()
    elif model_name == "ResNet18":
        model = ResNet(BasicBlock, [2, 2, 2, 2], in_ch=in_ch)
    elif model_name == "VGG11":
        model = vgg11()
    return model


class _TrainClassification:

    def __init__(self, city_name, model_name):
        self.model_name = model_name
        self.city_name = city_name

        cm_names = ["IS", "VEG", "SOIL", "WAT"]
        get_names = [
            #  0  1  2  3  4  5
            "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
            #  6  7  8  9 10 11
            "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
            # 12 13 14 15 16 17
            "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
        ]

        csv_fn = getCSVFn(city_name)
        sd = SamplesData()
        sd.addDLCSV(csv_fn, (21, 21), get_names, x_deal)
        win_size = (21, 21)
        read_size = (21, 21)

        model = getTorchInitModel(model_name, 4, len(get_names))

        raster_fns = getRasterFileNames(city_name)

        torch_mod = TorchModel()
        torch_mod.filename = r"F:\Week\20240707\Data\model3\model"
        torch_mod.map_dict = {
            "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
            "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
        }
        torch_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        torch_mod.model = model
        torch_mod.criterion = nn.CrossEntropyLoss()
        torch_mod.win_size = win_size
        torch_mod.read_size = read_size
        torch_mod.epochs = 102
        torch_mod.train_filters.extend([("city", "==", city_name)])
        torch_mod.test_filters.extend([("city", "==", city_name)])
        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts()
        torch_mod.n_epoch_save = 50

        print(torch_mod.model)

        self.torch_mod = torch_mod
        self.raster_fns = raster_fns

    def train(self, sw=None):
        dirnames = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp\dirnames.txt"

        dfn = DirFileName(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods")
        td = TimeDirectory(dfn.fn())
        td.initLog()
        td.log(td.time_dfn.dirname)
        if sw is not None:
            sw.write("training td.time_dfn.dirname")
            sw.write(td.time_dfn.dirname)
        with open(dirnames, "w", encoding="utf-8") as f:
            f.write(td.time_dfn.dirname)
        td.buildWriteText("city_name_{}.txt".format(self.city_name)).write(self.city_name)

        save_model_fmt = td.fn(self.model_name + "-" + self.city_name + "_{}.pth")
        print("save_model_fmt", save_model_fmt)

        self.torch_mod.save_model_fmt = save_model_fmt

        line_sw = td.buildWriteText("training-log.txt", "a")
        to_list = []
        td = td

        def func_field_record_save(field_records: FieldRecords):
            line = field_records.line
            for k in line:
                line_sw.write("| {}:{} ".format(k, line[k]), end="")
            line_sw.write("|")
            to_list.append(line.copy())

        self.torch_mod.func_field_record_save = func_field_record_save

        self.torch_mod.train()

        td.saveJson("training-log.json", to_list)

    def imdc(self, mod_fn):
        self.torch_mod.imdc(self.raster_fns, data_deal=x_deal, mod_fn=mod_fn)


def main():
    def func1():
        city_names = ["qd", "bj", "cd"]
        model_names = ["AlexNet", "ConvNeXt", "GoogLeNet", "ResNet18", "VGG11", ]
        tc = _TrainClassification(city_names[2], model_names[3])
        tc.train()

    def func2():
        get_names = [
            #  0  1  2  3  4  5
            "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2",
            #  6  7  8  9 10 11
            "AS_VV", "AS_VH", "AS_C11", "AS_C22", "AS_H", "AS_Alpha",
            # 12 13 14 15 16 17
            "DE_VV", "DE_VH", "DE_C11", "DE_C22", "DE_H", "DE_Alpha",
        ]

        city_name = "qd"
        csv_fn = getCSVFn("qd")
        sd = SamplesData()
        sd.addDLCSV(
            csv_fn, (21, 21), get_names, x_deal,
            grs={"qd": SHH2Config.QD_GR(), "bj": SHH2Config.BJ_GR(), "cd": SHH2Config.CD_GR(), }
        )

        win_size = (21, 21)
        read_size = (21, 21)

        model = ResNet(BasicBlock, [2, 2, 2, 2], in_ch=len(get_names), num_classes=4)

        torch_mod = TorchModel()
        torch_mod.filename = r"F:\Week\20240721\Data\model.hm"
        torch_mod.map_dict = {
            "IS": 0, "VEG": 1, "SOIL": 2, "WAT": 3,
            "IS_SH": 0, "VEG_SH": 1, "SOIL_SH": 2, "WAT_SH": 3,
        }
        torch_mod.color_table = {1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 255, 0), 4: (0, 0, 255), }
        torch_mod.model = model
        torch_mod.criterion = nn.CrossEntropyLoss()
        torch_mod.win_size = win_size
        torch_mod.read_size = read_size
        torch_mod.epochs = 2
        torch_mod.train_filters.extend([("city", "==", city_name)])
        torch_mod.test_filters.extend([("city", "==", city_name)])
        torch_mod.sampleData(sd)
        torch_mod.samples.showCounts()
        torch_mod.n_epoch_save = -1

        torch_mod.save_model_fmt = r"F:\Week\20240721\Data\model{}.pth"
        torch_mod.train()

        torch_mod.imdc(SHH2Config.QD_ENVI_FN, data_deal=x_deal, mod_fn=None, read_size=(1200, -1),is_save_tiles=True)

    return func2()


def run(city_name, model_name, is_train):
    sw = SRTWriteText(r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp\run1.txt", "a")
    sw.write(city_name, model_name, is_train)
    tc = _TrainClassification(city_name, model_name)

    if is_train:
        tc.train(sw)

    else:
        dirnames = r"F:\ProjectSet\Shadow\Hierarchical\GDDLMods\Temp\dirnames.txt"
        with open(dirnames, "r", encoding="utf-8") as f:
            dirname = f.read()
        dirname = dirname.strip()
        sw.write(dirname)
        fns = filterFileEndWith(dirname, ".pth")
        for fn in fns:
            if "1.pth" in fn:
                continue
            tc.imdc(fn)

    sw.write()


def funcs():
    city_names = ["qd", "bj", "cd"]
    model_names = ["AlexNet", "ConvNeXt", "GoogLeNet", "ResNet18", "VGG11", ]
    cmd_line = '''python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL2 import main; main()"'''
    # for model_name in model_names:


if __name__ == "__main__":
    # run("cd", "ResNet18", True)
    main()
    r"""
    python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL2 import main; main()"
    python -c "import sys; sys.path.append(r'F:\PyCodes'); from Shadow.Hierarchical.SHH2DL2 import run; run()"
    """
