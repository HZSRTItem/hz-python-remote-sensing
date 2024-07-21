# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : PytorchModelTraining.py
@Time    : 2023/6/22 16:39
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PytorchGeo of PytorchModelTraining
-----------------------------------------------------------------------------"""
import os
from shutil import copyfile

import numpy as np
import torch
import torchvision
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from SRTCodes.ModelTraining import CategoryTraining, RegressionTraining, Training, ConfusionMatrix
from SRTCodes.Utils import FieldRecords, TimeRunning, Jdt


def pytorchModelCodeString(model):
    model_str = ""
    if model is not None:
        try:
            import inspect
            model_str = inspect.getsource(model.__class__)
        except Exception as ex:
            model_str = str(ex)
    return model_str


def dataLoaderString(data_loader):
    model_str = {}
    if data_loader is not None:
        model_str = {"len": len(data_loader)}
    return model_str


class PytorchTraining(Training):

    def __init__(self, epochs=10, device=None, n_test=100):
        Training.__init__(self, None, None)
        self.epochs = epochs
        self.device = device
        self.n_test = n_test

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None
        self.loss = None
        self.scheduler = None

        if self._log is None:
            return

        self._log.addField("Epoch", "int")
        self._log.addField("Batch", "int")
        self._log.addField("Loss", "float")
        self._log.printOptions(print_float_decimal=3)

        return

    def toDict(self):
        to_dict_1 = super(PytorchTraining, self).toDict()
        to_dict_1["model"] = pytorchModelCodeString(self.model)
        to_dict_1["models"] = [pytorchModelCodeString(model) for model in self.models]

        to_dict = {
            **to_dict_1,
            "epochs": self.epochs,
            "device": self.device,
            "n_test": self.n_test,
            "train_loader": dataLoaderString(self.train_loader),
            "test_loader": dataLoaderString(self.test_loader),
            "optimizer": str(self.optimizer),
            "criterion": pytorchModelCodeString(self.criterion),
            "scheduler": str(self.scheduler),
        }

        return to_dict

    def saveModelCodeFile(self, model_code_file):
        copyfile(model_code_file, os.path.join(self.model_dir, os.path.split(model_code_file)[1]))

    def trainLoader(self, ds: Dataset, batch_size=128, shuffle=True, sampler=None, batch_sampler=None, num_workers=0):
        self.train_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                       batch_sampler=batch_sampler, num_workers=num_workers)

    def testLoader(self, ds: Dataset, batch_size=128, shuffle=True, sampler=None, batch_sampler=None, num_workers=0):
        self.test_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                      batch_sampler=batch_sampler, num_workers=num_workers)

    def addCriterion(self, criterion):
        self.criterion = criterion

    def addOptimizer(self, optim_func="adam", lr=0.001, eps=0.00001, optimizer=None, ):
        if optim_func == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=eps)
        else:
            self.optimizer = optimizer

    def addScheduler(self, scheduler):
        self.scheduler = scheduler

    def _initTrain(self):
        if self.model is None:
            raise Exception("Model can not find.")
        if self.criterion is None:
            raise Exception("Criterion can not find.")
        if self.optimizer is None:
            raise Exception("Optimizer can not find.")
        self.model.to(self.device)
        self.criterion.to(self.device)

    def _printModel(self):
        print("model:\n", self.model)
        print("criterion:\n", self.criterion)
        print("optimizer:\n", self.optimizer)

    def lossDeal(self, loss=None):
        if loss is None:
            loss = self.loss
        # L1_reg = 0  # 为Loss添加L1正则化项
        # for param in model.parameters():
        #     L1_reg += torch.sum(torch.abs(param))
        # loss += 0.001 * L1_reg  # lambda=0.001
        return loss

    def logBefore(self, batch, epoch):
        self._log.updateField("Epoch", epoch + 1)
        self._log.updateField("Batch", batch + 1)
        self._log.updateField("Loss", self.loss.item())
        # model_name = "{0}_{1}.pth".format(self.model_name, str(len(self._log.field_datas)))
        model_name = "{0}_Epoch{1}_Batch{2}.pth".format(self.model_name, batch, epoch)
        self._log.updateField("ModelName", model_name)
        return model_name

    def saveModel(self, model_name, *args, **kwargs):
        mod_fn = os.path.join(self.model_dir, model_name)
        torch.save(self.model.state_dict(), mod_fn)
        return mod_fn

    def train(self, batch_save=False, epoch_save=True, *args, **kwargs):
        self.initTrain()

        # for epoch in range(self.epochs):
        #     self.model.train()
        #     for batchix, (x, y) in enumerate(self.train_loader):
        #         x, y = x.to(self.device), y.to(self.device)
        #         x, y = x.float(), y.float()
        #         logts = self.model(x)  # 模型训练
        #         self.loss = self.criterion(logts, y)  # 损失函数
        #         self.loss = self.lossDeal(self.loss)  # loss处理
        #         self.optimizer.zero_grad()  # 梯度清零
        #         self.loss.backward()  # 反向传播
        #         self.optimizer.step()  # 优化迭代
        #
        #         # 测试 ------------------------------------------------------------------
        #         if self.test_loader is not None:
        #             if batchix % self.n_test == 0:
        #                 self.testAccuracy()
        #                 modname = self.log(batchix, epoch)
        #                 if batch_save:
        #                     self.saveModel(modname)
        #
        #     print("-" * 73)
        #     self.testAccuracy()
        #     modname = self.log(-1, epoch)
        #     modname = self.model_name + "_epoch_{0}.pth".format(epoch)
        #     print("*" * 73)
        #
        #     if epoch_save:
        #         self.saveModel(modname)

        for epoch in range(self.epochs):

            for batchix, (x, y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                x, y = x.float(), y.long()

                self.model.train()

                logts = self.model(x)
                self.loss = self.criterion(logts, y)
                self.loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()

                self.batchTAcc(batch_save, batchix, epoch)

            self.epochTAcc(epoch, epoch_save)

            if self.scheduler is not None:
                self.scheduler.step()

    def initTrain(self):
        self._initTrain()
        self._printModel()
        self._log.saveHeader()

    def log(self, batch, epoch):
        model_name = self.logBefore(batch, epoch)
        return model_name

    def batchTAcc(self, batch_save, batchix, epoch):
        if self.test_loader is not None:
            if batchix % self.n_test == 0:
                self.testAccuracy()
                modname = self.log(batchix, epoch)
                if batch_save:
                    self.saveModel(modname)

    def epochTAcc(self, epoch, epoch_save):
        print("-" * 80)
        self.testAccuracy()
        self.log(-1, epoch)
        modname = self.model_name + "_epoch_{0}.pth".format(epoch)
        if epoch_save:
            mod_fn = self.saveModel(modname)
            print("MODEL:", mod_fn)
        print("*" * 80)


class PytorchCategoryTraining(CategoryTraining, PytorchTraining):

    def __init__(self, model_dir=None, model_name="model", n_category=2, category_names=None,
                 epochs=10, device=None, n_test=100):
        """ Pytorch Training init model_dir=None, model_name="model", n_category=2, category_names=None
        /
        :param n_category: number of category
        :param model_dir: save model directory
        :param category_names: category names
        :param epochs: number of epochs
        :param device: device of cpu or cuda
        :param n_test: How many samples are tested every
        :param model_name: model name
        """
        CategoryTraining.__init__(self, model_dir=model_dir, model_name=model_name, n_category=n_category,
                                  category_names=category_names)
        PytorchTraining.__init__(self, epochs=epochs, device=device, n_test=n_test)
        self.timeModelDir()
        self._log.printOptions(print_type="keyword", print_field_names=["Epoch", "Batch", "Loss", "OATest"])

    def logisticToCategory(self, logts):
        return logts

    def testAccuracy(self, deal_y_func=None):
        self.test_cm.clear()
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                x = x.float()
                if deal_y_func is not None:
                    y = deal_y_func(y)
                y = y.numpy()
                logts = self.model(x)
                y1 = self.logisticToCategory(logts)
                self.test_cm.addData(y, y1)
        self.model.train()
        return self.test_cm.OA()

    def log(self, batch, epoch):
        model_name = self.logBefore(batch, epoch)

        if self.train_loader is not None:
            self.updateLogTrainCM()
        if self.test_loader is not None:
            self.updateLogTestCM()

        self._log.saveLine()
        self._log.print(is_to_file=True)
        self._log.newLine()
        return model_name


class TorchTraining:

    def __init__(self):
        self.model = nn.Sequential()
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = 100
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.n_test = 10
        self.train_loader = None
        self.test_loader = None
        self.batch_save = False
        self.epoch_save = True
        self.save_model_fmt = None
        self.n_epoch_save = 1
        self.win_size = None

        self._optimizer = None
        self._scheduler = None
        self.loss = None

        self.train_cm = ConfusionMatrix()
        self.test_cm = ConfusionMatrix()

        self.field_records = FieldRecords()
        self.field_records.addFields("Epoch", "Batch", "Loss", "Accuracy")

        def func_logit_category(model, x: torch.Tensor):
            logit = model(x)
            y = torch.argmax(logit, dim=1) + 1
            return y

        self.func_epoch = None
        self.func_xy_deal = None
        self.func_batch = None
        self.func_loss_deal = None
        self.func_y_deal = lambda y: y + 1
        self.func_logit_category = func_logit_category
        self.func_print = print
        self.func_field_record_save = None

        self.time_run = None

    def initCM(self, n=0, cnames=None):
        self.test_cm = ConfusionMatrix(n, cnames)
        self.field_records.addFields("OA Test")
        self.field_records.addFields("Kappa Test")
        for name in self.test_cm.CNAMES():
            self.field_records.addFields("{} UA Test".format(name))
            self.field_records.addFields("{} PA Test".format(name))
        self.train_cm = ConfusionMatrix(n, cnames)
        self.field_records.addFields("OA Train")
        self.field_records.addFields("Kappa Train")
        for name in self.train_cm.CNAMES():
            self.field_records.addFields("{} UA Train".format(name))
            self.field_records.addFields("{} PA Train".format(name))

    def trainLoader(self, ds: Dataset, batch_size=128, shuffle=True, sampler=None, batch_sampler=None, num_workers=0):
        self.train_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                       batch_sampler=batch_sampler, num_workers=num_workers)
        n = self.epochs * len(self.train_loader)
        self.time_run = TimeRunning(n)

    def testLoader(self, ds: Dataset, batch_size=128, shuffle=True, sampler=None, batch_sampler=None, num_workers=0):
        self.test_loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
                                      batch_sampler=batch_sampler, num_workers=num_workers)

    def optimizer(self, optim_cls=optim.Adam, *args, **kwargs):
        """ optim.Adam lr=0.001, eps=0.0000001 """
        self._optimizer = optim_cls(self.model.parameters(), *args, **kwargs)

    def scheduler(self, scheduler_cls=optim.lr_scheduler.StepLR, *args, **kwargs):
        """ optim.lr_scheduler.StepLR step_size=20, gamma=0.6, last_epoch=-1 """
        self._scheduler = scheduler_cls(self._optimizer, *args, **kwargs)

    def train(self, *args, **kwargs):
        self.model.to(self.device)
        self.criterion.to(self.device)

        if self._optimizer is None:
            self.optimizer(lr=0.001, eps=0.0000001)

        if self.time_run is None:
            n = self.epochs * len(self.train_loader)
            self.time_run = TimeRunning(n)
        self.time_run.start()

        for epoch in range(self.epochs):

            if self.func_epoch is not None:
                self.func_epoch()

            for batch, (x, y) in enumerate(self.train_loader):
                # x, y = x.to(self.device), y.to(self.device)
                if self.win_size is None:
                    self.win_size = int(x.shape[2]), int(x.shape[3])

                if self.func_xy_deal is not None:
                    x, y = self.func_xy_deal(x, y)

                if self.func_batch is not None:
                    self.func_batch()

                self.model.train()

                logts = self.model(x)
                self.loss = self.criterion(logts, y)
                if self.func_loss_deal is not None:
                    self.loss = self.func_loss_deal(self.loss)

                self._optimizer.zero_grad()
                self.loss.backward()
                self._optimizer.step()

                self.testing(epoch, batch)

            self.testing(epoch, -1)

            if self._scheduler is not None:
                self._scheduler.step()

    def testing(self, epoch, batch):
        epoch += 1
        batch += 1
        self.time_run.add(is_show=False)

        self.field_records.clearLine()
        self.field_records.line["Epoch"] = epoch
        self.field_records.line["Batch"] = batch
        self.field_records.line["Loss"] = float(self.loss.item())

        def _print(_acc):
            if self.func_print is None:
                return
            self.func_print("\r", end="")
            if batch == 0:
                self.func_print("-" * 100)
            self.func_print("+ Epoch:", "{:<6d}".format(epoch), end=" ")
            self.func_print("Batch:", "{:<6d}".format(batch), end=" ")
            self.func_print("Loss:", "{:<12.6f}".format(float(self.loss.item())), end=" ")
            self.func_print("Accuracy:", "{:>6.3f}".format(_acc), end="   ")
            self.func_print(self.time_run.fmt, end="\n")
            if batch == 0:
                if (self.save_model_fmt is not None) and self.epoch_save:
                    mod_fn = self.save_model_fmt.format(str(epoch))

                    def _save_epoch():
                        torch.save(self.model.state_dict(), mod_fn)
                        self.func_print("- MODEL: {}".format(mod_fn), end="\n")

                    if self.n_epoch_save == -1:
                        if epoch == self.epochs:
                            _save_epoch()
                    else:
                        if epoch == 1:
                            _save_epoch()
                        elif epoch % self.n_epoch_save == 0:
                            _save_epoch()

                self.func_print("*" * 100)
            if (self.save_model_fmt is not None) and (batch % self.n_test == 0) and self.batch_save:
                mod_fn = self.save_model_fmt.format(str(epoch) + "-" + str(batch))
                torch.save(self.model.state_dict(), mod_fn)
                self.func_print("MODEL: {}".format(mod_fn), end="\n")

        acc = -1
        if self.test_loader is not None:
            if batch == 0:
                acc = self.accuracy()
                _print(acc)
            elif batch % self.n_test == 0:
                acc = self.accuracy()
                _print(acc)
            else:
                acc = -1
                if self.func_print is not None:
                    self.func_print("\r{}".format(self.time_run.fmt), end="")

        self.field_records.line["Accuracy"] = float(acc)
        self.field_records.addLine()
        if self.func_field_record_save is not None:
            self.func_field_record_save(self.field_records)

    def accuracy(self):

        def _accuracy(_data_loader, _cm, update_name):

            _cm.clear()
            n, n_total = 0, 0
            if _data_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for i, (x, y) in enumerate(self.test_loader):
                        # x = x.to(self.device).float()
                        # y = y.numpy()
                        if self.func_y_deal is not None:
                            y = self.func_y_deal(y)
                        y1 = self.func_logit_category(self.model, x)
                        # y1 = y1.cpu().numpy()
                        if _cm is not None:
                            _cm.addData(y, y1)
                        else:
                            n += np.sum((y == y1) * 1)
                            n_total += len(y)
                self.model.train()

            if _cm is not None:
                _acc = _cm.OA()
                self.field_records.line["OA {}".format(update_name)] = _cm.OA()
                self.field_records.line["Kappa {}".format(update_name)] = _cm.getKappa()
                for name in self.test_cm.CNAMES():
                    self.field_records.line["{} UA {}".format(name, update_name)] = _cm.UA(name)
                    self.field_records.line["{} PA {}".format(name, update_name)] = _cm.PA(name)
            else:
                _acc = n * 1.0 / n_total

            return _acc

        _accuracy(self.train_loader, self.train_cm, "Train")
        acc = _accuracy(self.test_loader, self.test_cm, "Test")

        return acc

    def imdc(self, data, is_jdt=True):
        self.model.eval()
        imdc = torchDataPredict(
            data=data, win_size=self.win_size,
            func_predict=lambda x: self.func_logit_category(self.model, x),
            device=self.device, is_jdt=is_jdt
        )
        self.model.train()
        return imdc


def torchDataPredict(data, win_size, func_predict, data_deal=None, device="cuda", is_jdt=True):
    imdc = torch.zeros(data.shape[1], data.shape[2], device=device)
    data = data.to(device)
    if data_deal is not None:
        data = data_deal(data)
    data = data.unfold(1, win_size[0], 1).unfold(2, win_size[1], 1)
    row_start, column_start = int(win_size[0] / 2), int(win_size[1] / 2)
    jdt = Jdt(data.shape[1], "Torch Data Predict").start(is_jdt=is_jdt)
    for i in range(data.shape[1]):
        x_data = data[:, i]
        x_data = torch.transpose(x_data, 0, 1)
        y = func_predict(x_data)
        imdc[row_start + i, column_start: column_start + data.shape[2]] = y
        jdt.add(is_jdt=is_jdt)
    jdt.end(is_jdt=is_jdt)
    imdc = imdc.cpu().numpy()
    return imdc


class PytorchRegressionTraining(RegressionTraining, PytorchTraining):

    def __init__(self, model_dir=None, model_name="model", epochs=10, device=None, n_test=100):
        RegressionTraining.__init__(self, model_dir=model_dir, model_name=model_name)
        PytorchTraining.__init__(self, epochs=epochs, device=device, n_test=n_test)
        self.timeModelDir()
        self._log.printOptions(print_type="keyword", print_field_names=["Epoch", "Batch", "Loss", "MSETest"])

    def testAccuracy(self):
        self.test_mse.clear()
        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(self.test_loader):
                x = x.to(self.device)
                x = x.float()
                y = y.numpy()
                logts = self.model(x)  # [bs, 10]
                self.test_mse.add(y, logts.cpu().numpy())
        return self.test_mse.MSE()

    def log(self, batch, epoch):
        model_name = self.logBefore(batch, epoch)

        if self.train_loader is not None:
            self.updateLogTrainMSE()
        if self.test_loader is not None:
            self.updateLogTestMSE()

        self._log.saveLine()
        self._log.print(is_to_file=True)
        self._log.newLine()
        return model_name


def loadMNISTDataset():
    def func(train):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = torchvision.datasets.MNIST(
            r"G:\Dataset\MyMNIST",
            train=train,
            download=True,
            transform=transform,
        )
        return ds

    train_ds = func(True)
    test_ds = func(False)
    return train_ds, test_ds


def main():
    class _DLDS(Dataset):

        def __init__(self):
            self.data = torch.rand(10000, 3, 3, 3)
            self.y = torch.randint(0, 3, (10000,))

        def __getitem__(self, item):
            return self.data[item], self.y[item]

        def __len__(self):
            return 10000

    torch_training = TorchTraining()
    torch_training.model = nn.Sequential(
        nn.Conv2d(3, 6, 3, 1, 1),
        nn.Flatten(start_dim=1),
        nn.Linear(54, 3),
    )
    torch_training.criterion = nn.CrossEntropyLoss()
    torch_training.epochs = 100
    torch_training.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_training.n_test = 50
    torch_training.trainLoader(_DLDS(), 32)
    torch_training.testLoader(_DLDS(), 32)
    torch_training.optimizer(optim.Adam, lr=0.001, eps=0.000001)
    torch_training.scheduler(optim.lr_scheduler.StepLR, step_size=20, gamma=0.6, last_epoch=-1)
    torch_training.initCM(3)
    torch_training.batch_save = False
    torch_training.epoch_save = True
    torch_training.save_model_fmt = r"F:\Week\20240707\Data\model{}.pth"
    torch_training.train()

    return


if __name__ == "__main__":
    main()
