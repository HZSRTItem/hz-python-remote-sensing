# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : ShadowRun.py
@Time    : 2023/12/25 9:41
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2023, ZhengHan. All rights reserved.
@Desc    : PyCodes of ShadowRun
-----------------------------------------------------------------------------"""

import sys

sys.path.append(r"F:\PyCodes")

from Shadow.ShadowMainBeiJing import ShadowMainBJ
from Shadow.ShadowMainChengDu import ShadowMainCD
from Shadow.ShadowMainQingDao import ShadowMainQD


class ShadowRun:

    def __init__(self):
        self.run_name = ""
        self.sm_qd = None
        self.sm_bj = None
        self.sm_cd = None

    def run(self):
        self.run_name = sys.argv[1].upper()
        self.runQD_shadowTraining()
        self.runBJ_shadowTraining()
        self.runCD_shadowTraining()

    def runQD_shadowTraining(self):
        if self.run_name != "QD":
            return None
        if self.sm_cd is None:
            self.sm_qd = ShadowMainQD()
        self.sm_qd.shadowTraining()

    def runBJ_shadowTraining(self):
        if self.run_name != "BJ":
            return None
        if self.sm_bj is None:
            self.sm_bj = ShadowMainBJ()
        self.sm_bj.shadowTraining()

    def runCD_shadowTraining(self):
        if self.run_name != "CD":
            return None
        if self.sm_cd is None:
            self.sm_cd = ShadowMainCD()
        self.sm_cd.shadowTraining()

    def usage(self):
        print("sh_run [QD|BJ|CD]")


def main():
    print(__file__)
    shr = ShadowRun()
    if len(sys.argv) == 1:
        shr.usage()
        return
    shr.run()


if __name__ == "__main__":
    # python F:\PyCodes\Shadow\ShadowRun.py %*
    main()
