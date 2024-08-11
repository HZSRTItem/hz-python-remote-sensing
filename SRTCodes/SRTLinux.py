# _*_ coding:utf-8 _*_
r"""----------------------------------------------------------------------------
@File    : SRTLinux.py
@Time    : 2024/8/4 10:05
@Author  : Zheng Han 
@Contact : tourensong@gmail.com
@License : (C)Copyright 2024, ZhengHan. All rights reserved.
@Desc    : PyCodes of SRTLinux
https://github.com/riag/pywslpath/tree/master
-----------------------------------------------------------------------------"""

import os
import sys

from SRTCodes.DEFINE import StrOrBytesPath

# _LINUX_TYPE = "UBUNTU"
_LINUX_TYPE = "NONE"


def W2LF(path:StrOrBytesPath):
    if _LINUX_TYPE == "UBUNTU":
        if path[1] == ":":
            path = "/mnt/" + path[0].lower() + path[2:]
        path = path.replace("\\", "/")
        return path
    return path


def main():
    dirname = W2LF(r"F:\PyCodes\SRTCodes")
    for f in os.listdir(dirname):
        if f.endswith(".py"):
            print(f)
            w2lfFindStr(os.path.join(dirname, f))
    pass


def w2lfFindStr(fn=None):
    if fn is None:
        if len(sys.argv) >= 2:
            fn = sys.argv[1]
        else:
            raise Exception("Can not find filename.")

    _list = ["\"{}:".format(s) for s in "QWERTYUIOPSDFGHJKLZXCVNM"]
    _list.sort()
    with open(fn, "r", encoding="utf-8") as fr:
        for i, line in enumerate(fr):
            line: str
            for s in _list:
                if s in line:
                    print("{}. {}:{}".format(i + 1, fn, line.strip()))

if __name__ == "__main__":
    main()
    r"""
    
    python -c "import sys; sys.path.append(r'F:\PyCodes'); from SRTCodes.SRTLinux import w2lfFindStr; w2lfFindStr()"
    """
