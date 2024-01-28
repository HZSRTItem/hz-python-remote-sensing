# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : XuLin.py
@Time    : 2024/1/13 9:38
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of XuLin
-----------------------------------------------------------------------------"""
import torch
from torch import nn, optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d()
        self.bn = nn.BatchNorm2d()

        # ...

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)

        # ...

        return x


class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.sigmoid_act = nn.Sigmoid()

    def forward(self, x, y, lamd=1):
        p = self.sigmoid_act(x)
        y = y.view((y.size(0), 1))
        loss = -torch.mean(lamd * (y * torch.log(p) + (1 - y) * torch.log(1 - p)))
        return loss


def train():
    epochs = 100
    train_loader = None  # 看一下 train loader
    test_loader = None

    n_test = 10  # 几次循环做一次测试

    device = "cuda"

    model = Net()
    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 学习率

    for epoch in range(epochs):

        for batchix, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            x, y = x.float(), y.float()

            logts = model(x)  # 模型训练
            loss = criterion(logts, y)  # 损失函数

            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播
            optimizer.step()  # 优化迭代

            # 测试 ------------------------------------------------------------------
            if test_loader is not None:
                if batchix % n_test == 0:
                    # 测试代码
                    pass

        print("epoch:", epoch)
        # 测试
        mod_fn = ""  # 设计一个明
        torch.save(model.state_dict(), mod_fn)



def main():

    train()
    pass


if __name__ == "__main__":
    main()
