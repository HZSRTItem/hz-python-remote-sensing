# 这是一个示例 Python 脚本。
# 按 Ctrl+F5 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import os.path
import shutil
import sys
from datetime import datetime




def print_hi(name):
    # 在下面的代码行中使用断点来调试脚本。
    print(f'Hi, {name}')  # 按 F9 切换断点。


# 按间距中的绿色按钮以运行脚本。


if __name__ == '__main__':
    if sys.argv[-1] == "back":
        shutil.copytree(r"F:\PyCodes", os.path.join(
            r"F:\back", datetime.now().strftime("%Y-%m-%d %H-%M-%S"), "PyCodes"))
    d = []
    for i in range(10):
        d.append(i + 1)



# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
