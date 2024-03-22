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

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader

from SRTCodes.ModelTraining import CategoryTraining, RegressionTraining, Training


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

        self._log.addField("Epoch", "int")
        self._log.addField("Batch", "int")
        self._log.addField("Loss", "float")
        self._log.printOptions(print_float_decimal=3)

        return

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
        self._initTrain()
        self._printModel()
        self._log.saveHeader()

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


def main():
    pass


if __name__ == "__main__":
    main()
