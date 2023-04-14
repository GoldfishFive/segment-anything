import os, sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel


from widgets.MainWindow import MainWindow

def main():
    app = QApplication(sys.argv)
    window = MainWindow()  # 创建PyQt设计器的窗体对象
    window.show()  # 显示窗体
    sys.exit(app.exec_())  # 程序关闭时退出进程


if __name__ == '__main__':
    main()
