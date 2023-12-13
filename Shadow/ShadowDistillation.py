# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowDistillation.py
@Time    : 2023/11/11 14:05
@Author  : Zheng Han 
@Contact : hzsongrentou1580@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowDistillation
-----------------------------------------------------------------------------"""
import os
from typing import Callable, List, Optional, Type, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from SRTCodes.GDALRasterClassification import GDALRasterPrediction
from SRTCodes.GDALRasterIO import GDALRaster
from SRTCodes.PytorchModelTraining import PytorchCategoryTraining
from SRTCodes.SRTData import SRTDataset
from SRTCodes.Utils import DirFileName

SD_BJ_DFN = DirFileName(r"F:\ProjectSet\Shadow\DeepLearn\BeiJing")


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation, )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ShadowTeacherBasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ShadowTeacherResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[ShadowTeacherBasicBlock]],
            in_channels: int,
            n_blocks: List[int],
            strides: Tuple[int, int, int, int] = (1, 2, 2, 2),
            num_classes: int = 1000,
            is_to_loss=False,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        self.is_to_loss = is_to_loss

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
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, n_blocks[0], stride=strides[0])
        self.to_loss_conv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.layer2 = self._make_layer(block, 128, n_blocks[1], stride=strides[1],
                                       dilate=replace_stride_with_dilation[0])
        self.to_loss_conv2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1)

        self.layer3 = self._make_layer(block, 256, n_blocks[2], stride=strides[2],
                                       dilate=replace_stride_with_dilation[1])
        self.to_loss_conv3 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1)

        self.layer4 = self._make_layer(block, 512, n_blocks[3], stride=strides[3],
                                       dilate=replace_stride_with_dilation[2])
        self.to_loss_conv4 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1)

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
                # if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                #     nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                if isinstance(m, ShadowTeacherBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[ShadowTeacherBasicBlock]],
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                            norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer, ))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        to_loss1 = self.to_loss_conv1(x)
        x = self.layer2(x)
        to_loss2 = self.to_loss_conv2(x)
        x = self.layer3(x)
        to_loss3 = self.to_loss_conv3(x)
        x = self.layer4(x)
        to_loss4 = self.to_loss_conv4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, [to_loss1, to_loss2, to_loss3, to_loss4]

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)


class ShadowStudentBlock(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_layers: int,
            stride: int = 1,
            is_to_loss=False,
            norm_layer: Optional[Callable[..., nn.Module]] = None, ):
        super(ShadowStudentBlock, self).__init__()
        self.is_to_loss = is_to_loss

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.layers = [self.buildLayer(in_channels, out_channels, stride, 3, norm_layer, 1)]
        for i in range(1, n_layers):
            self.layers.append(self.buildLayer(out_channels, out_channels, 1, 3, norm_layer, 1))

        self.to_loss_conv = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=1, stride=1)

    def buildLayer(self, in_channels, out_channels, stride, kernel_size, norm_layer, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=1),
            norm_layer(out_channels),
            nn.ReLU())

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)

        to_loss = None
        if self.is_to_loss:
            to_loss = self.to_loss_conv(x)

        return x, to_loss


class ShadowStudentNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[ShadowStudentBlock]],
            in_channels: int,
            layers: List[int],
            strides: Tuple[int, int, int, int] = (1, 1, 1, 1),
            num_classes: int = 1000,
            is_to_loss=False,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None, ):
        super(ShadowStudentNet, self).__init__()
        self.is_to_loss = is_to_loss
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.conv1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(in_channels * 2)
        self.relu = nn.ReLU()

        self.channels = in_channels * 2
        self.layer1 = self._make_layer(block, layers[0], strides[0])
        self.layer2 = self._make_layer(block, layers[1], strides[1])
        self.layer3 = self._make_layer(block, layers[2], strides[2])
        self.layer4 = self._make_layer(block, layers[3], strides[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, n_layers, stride):
        layer = block(self.channels, self.channels * 2, n_layers, stride=stride,
                      norm_layer=self._norm_layer, is_to_loss=self.is_to_loss)
        self.channels *= 2
        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x, to_loss1 = self.layer1(x)
        x, to_loss2 = self.layer2(x)
        x, to_loss3 = self.layer3(x)
        x, to_loss4 = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, [to_loss1, to_loss2, to_loss3, to_loss4]


class ShadowDistillationNet(nn.Module):

    def __init__(
            self,
            in_channels,
            tea_n_blocks=(2, 2, 2, 2),
            stu_n_blocks=(1, 1, 1, 1),
            strides=(1, 1, 1, 2),
            is_to_loss=True
    ):
        super(ShadowDistillationNet, self).__init__()

        self.teacher_mod = ShadowTeacherResNet(ShadowTeacherBasicBlock, in_channels, list(tea_n_blocks),
                                               strides=strides, num_classes=1, is_to_loss=is_to_loss)
        self.student_mod = ShadowStudentNet(ShadowStudentBlock, in_channels, list(stu_n_blocks),
                                            strides=strides, num_classes=1, is_to_loss=is_to_loss)

    def forward(self, x):
        out_teacher = self.teacher_mod(x)
        out_student = self.student_mod(x)
        return out_teacher, out_student


class ShadowDistillationLoss(nn.Module):

    def __init__(self, n_kl=4):
        super(ShadowDistillationLoss, self).__init__()
        self.kl_losses = [nn.KLDivLoss(reduction='batchmean') for _ in range(n_kl)]
        self.kl_small = 0.1

    def forward(self, out_teacher, out_student, y: torch.Tensor):
        y = y.view((y.size(0), 1))
        logit_tea, logit_stu = torch.sigmoid(out_teacher[0]), torch.sigmoid(out_student[0])
        loss_tea = -torch.mean((y * torch.log(logit_tea) + (1 - y) * torch.log(1 - logit_tea)))
        loss_stu = -torch.mean((y * torch.log(logit_stu) + (1 - y) * torch.log(1 - logit_stu)))
        loss = loss_tea + loss_stu
        for i, kl_loss in enumerate(self.kl_losses):
            x = torch.flatten(out_student[1][i], start_dim=1)
            x = F.log_softmax(x, dim=1)
            target = torch.flatten(out_teacher[1][i], start_dim=1)
            target = F.softmax(target, dim=1)
            kl = kl_loss(x, target)
            loss += self.kl_small * kl
        return loss


class SDDataSet(SRTDataset, Dataset):

    def __init__(self, is_test=False, spl_fn="", train_d_fn="", cname="CATEGORY"):
        super().__init__()

        self.spl_fn = spl_fn
        self.train_d_fn = train_d_fn
        self.df = pd.read_csv(self.spl_fn)
        is_tests = self.df["TEST"].values
        categorys = self.df[cname].values.tolist()
        self.addCategory("NOIS")
        self.addCategory("IS")
        self.addNPY(categorys, self.train_d_fn)
        self._isTest(is_test, is_tests)

    def _isTest(self, is_test, is_tests):
        datalist = []
        category_list = []
        for i in range(len(self.datalist)):
            if is_test:
                if is_tests[i] == 1:
                    datalist.append(self.datalist[i])
                    category_list.append(self.category_list[i])
            else:
                if is_tests[i] == 0:
                    datalist.append(self.datalist[i])
                    category_list.append(self.category_list[i])
        self.datalist = datalist
        self.category_list = category_list

    def __getitem__(self, index):
        return self.get(index)

    def get(self, index):
        x = self.datalist[index]
        x = (np.clip(x, 200, 1600) - 200) / (1600 - 200)
        y = self.category_list[index]
        return x, y


class SDPytorchTraining(PytorchCategoryTraining):

    def __init__(self, n_category, model_dir=None, category_names=None, epochs=10, device=None, n_test=100):
        super().__init__(n_category=n_category, model_dir=model_dir, category_names=category_names, epochs=epochs,
                         device=device, n_test=n_test)

    def logisticToCategory(self, logts):
        logts = torch.sigmoid(logts)
        logts = (logts.cpu().numpy().T[0] > 0.5) * 1
        return logts

    def lossDeal(self, loss=None):
        if loss is None:
            loss = self.loss
        # L1_reg = 0  # 为Loss添加L1正则化项
        # for param in model.parameters():
        #     L1_reg += torch.sum(torch.abs(param))
        # loss += 0.001 * L1_reg  # lambda=0.001
        return loss

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):
        self._initTrain()
        self._printModel()
        self._log.saveHeader()

        teacher_mod = self.models[0].to(self.device)
        student_mod = self.models[1].to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.float(), y.float()

                out_teacher = teacher_mod(x)
                out_student = student_mod(x)
                self.loss = self.criterion(out_teacher, out_student, y)  # 损失函数

                self.loss = self.lossDeal(self.loss)  # loss处理
                self.optimizer.zero_grad()  # 梯度清零
                self.loss.backward()  # 反向传播
                self.optimizer.step()  # 优化迭代

                # 测试 ------------------------------------------------------------------
                if self.test_loader is not None:
                    if batchix % self.n_test == 0:
                        self.testAccuracy()
                        modname = self.log(batchix, epoch)
                        if batch_save:
                            self.saveModel(modname)

            print("-" * 73)
            self.testAccuracy()
            modname = self.log(-1, epoch)
            print("*" * 73)

            if epoch_save:
                self.saveModel(modname)

    def testAccuracy(self):
        self.test_cm.clear()
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                x = x.float()
                y = y.numpy()
                logts = self.model(x)
                y1 = self.logisticToCategory(logts)
                self.test_cm.addData(y, y1)
        return self.test_cm.OA()


class JPZ5MXL21_GDALRasterPrediction(GDALRasterPrediction):

    def __init__(self, geo_fn):
        super(JPZ5MXL21_GDALRasterPrediction, self).__init__(geo_fn)
        self.device = "cuda:0"
        self.is_category = False
        self.number_pred = 15000

    def predict(self, x: np.ndarray, *args, **kwargs) -> np.ndarray:
        # y = np.ones(x.shape[0])
        n = x.shape[0]

        x = x.astype("float32")
        # x = np.clip(x, 0, 1000) / 1000.0
        x = (np.clip(x, 200, 1600) - 200) / (1600 - 200)
        x = torch.from_numpy(x)
        x = x.to(self.device)
        y = torch.zeros((n, 1), dtype=torch.float)
        y = y.to(self.device)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, n, self.number_pred):
                y_temp = self.model(x[i:i + self.number_pred, :])
                y[i:i + self.number_pred, :] = y_temp
            y = torch.sigmoid(y)
        y = y.cpu().numpy()
        y = y.T[0]
        if self.is_category:
            y = (y > 0.5) * 1

        return y

    def preDeal(self, row, column_start, column_end):
        d_row = self.d[:, row, column_start:column_end]
        return np.ones(d_row.shape[1], dtype="bool")


class SDBJ_Main:

    def __init__(self):
        self.this_dirname = self.mkdir(r"H:\JPZ\JPZ5MXL21")
        self.model_dir = self.mkdir(os.path.join(self.this_dirname, "Mods"))
        self.n_category = 2
        self.category_names = ["NOIS", "IS"]
        self.epochs = 30
        self.device = "cuda:0"
        self.n_test = 10
        self.csv_fn = SD_BJ_DFN.fn("Samples", "sh_bj_spl.csv")
        self.npy_fn = SD_BJ_DFN.fn("Samples", "sh_bj_spl.npy")
        self.test_ds = None
        self.train_ds = None
        self.win_size = 9

        self.teacher_mod = ShadowTeacherResNet(ShadowTeacherBasicBlock, 8, [2, 2, 2, 2], strides=(1, 1, 1, 2),
                                               num_classes=1, is_to_loss=True)
        self.student_mod = ShadowStudentNet(ShadowStudentBlock, 4, [1, 1, 1, 1], strides=(1, 1, 1, 2),
                                            num_classes=1, is_to_loss=True)
        self.criterion = ShadowDistillationLoss()

        self.geo_raster = r"F:\ProjectSet\Shadow\BeiJing\Image\4\BJ_SH4.vrt"

    def mkdir(self, dirname):
        dirname = os.path.abspath(dirname)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        return dirname

    def train(self):
        self.train_ds = SDDataSet(is_test=False, spl_fn=self.csv_fn, train_d_fn=self.npy_fn, cname="CATEGORY")
        self.test_ds = SDDataSet(is_test=True, spl_fn=self.csv_fn, train_d_fn=self.npy_fn, cname="CATEGORY")

        pytorch_training = SDPytorchTraining(
            n_category=self.n_category,
            model_dir=self.model_dir,
            category_names=self.category_names,
            epochs=self.epochs,
            device=self.device,
            n_test=self.n_test
        )

        pytorch_training.trainLoader(self.train_ds, batch_size=128, shuffle=True)
        pytorch_training.testLoader(self.test_ds, batch_size=128, shuffle=False)
        pytorch_training.addModel(self.teacher_mod)
        pytorch_training.addModel(self.student_mod)
        pytorch_training.addCriterion(self.criterion)
        pytorch_training.addOptimizer(lr=0.0005)
        pytorch_training.train()

    def imdc(self, raster_dirname):
        mod_dirname = "20231009H083734"
        imdc_dirname = os.path.join(self.model_dir, mod_dirname, "model_20_imdc1")
        if not os.path.isdir(imdc_dirname):
            os.mkdir(imdc_dirname)
        imdc_fn = os.path.join(self.model_dir, mod_dirname, "model_20_imdc1.tif")
        mod_fn = os.path.join(self.model_dir, mod_dirname, "model_20.pth")
        np_type = "int8"
        self.mod.load_state_dict(torch.load(mod_fn))
        self.mod.to(self.device)
        for f in os.listdir(raster_dirname):
            if os.path.splitext(f)[1] == '.tif':
                print(f)
                geo_raster = os.path.join(raster_dirname, f)
                imdc_fn = os.path.join(imdc_dirname, f)
                if os.path.isfile(imdc_fn):
                    print("RasterClassification: 100%")
                    continue
                grp = JPZ5MXL21_GDALRasterPrediction(geo_raster)
                grp.is_category = True
                grp.run(imdc_fn=imdc_fn, np_type=np_type, mod=self.mod,
                        spl_size=[self.win_size, self.win_size],
                        row_start=self.win_size + 6, row_end=-(self.win_size + 6),
                        column_start=self.win_size + 6, column_end=-(self.win_size + 6),
                        n_one_t=20000)

    def sampleNPY(self):
        # 使用CSV文件在影像中提取样本的数据
        spl_fn = self.csv_fn
        raster_fn = self.geo_raster
        train_d_fn = self.npy_fn
        spl_size = [self.win_size, self.win_size]
        df = pd.read_csv(spl_fn)
        gr = GDALRaster(raster_fn)
        d = np.zeros([len(df), gr.n_channels, spl_size[0], spl_size[1]])
        print(d.shape)
        for i in range(len(df)):
            x, y = df["X"][i], df["Y"][i]
            d[i, :] = gr.readAsArrayCenter(x, y, win_row_size=spl_size[0], win_column_size=spl_size[1],
                                           interleave="band", is_geo=True, is_trans=True)
            if i % 500 == 0:
                print(i)
        np.save(train_d_fn, d)


def main():
    tmp, y = torch.randn(32, 6, 7, 7), torch.randint(0, 2, (32, 1))

    teacher_mod = ShadowTeacherResNet(ShadowTeacherBasicBlock, tmp.shape[1], [2, 2, 2, 2], strides=(1, 1, 1, 2),
                                      num_classes=1, is_to_loss=True)
    student_mod = ShadowStudentNet(ShadowStudentBlock, tmp.shape[1], [1, 1, 1, 1], strides=(1, 1, 1, 2),
                                   num_classes=1, is_to_loss=True)

    criterion = ShadowDistillationLoss()

    out_teacher = teacher_mod(tmp)
    out_student = student_mod(tmp)

    loss = criterion(out_teacher, out_student, y=y)

    print(loss)

    ...


if __name__ == "__main__":
    main()
    #
    #
    # kl_loss = nn.KLDivLoss(reduction="batchmean")
    # # input should be a distribution in the log space
    # x = F.log_softmax(torch.randn(3, 5, 5, requires_grad=True), dim=1)
    # print(x.shape)
    # # Sample a batch of distributions. Usually this would come from the dataset
    # target = F.softmax(torch.rand(3, 5, 5), dim=1)
    # print(target.shape)
    # output = kl_loss(x, target)
    # print(output)

    # kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    # log_target = F.log_softmax(torch.rand(3, 5), dim=1)
    # output = kl_loss(input, log_target)
