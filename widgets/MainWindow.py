import os

from PyQt5.QtWidgets import QMainWindow
from ui.sam_labeltool import Ui_MainWindow


class MainWindow(Ui_MainWindow,QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.first_flag = True
        self.setWindowTitle('SAM_LabelTool')
        # 调整窗口大小
        self.resize(800, 600)


    def showXY(self, point):
        # crsSrc = QgsCoordinateReferenceSystem(4326)    # WGS 84
        # crsDest = QgsCoordinateReferenceSystem(32633)  # WGS 84 / UTM zone 33N
        # xform = QgsCoordinateTransform(crsSrc, crsDest, QgsProject.instance())
        # # 正向转换: src -> dest
        # pt1 = xform.transform(QgsPointXY(18,5))
        # print("Transformed point:", pt1)
        # 逆向转换: dest -> src
        # pt2 = xform.transform(point, QgsCoordinateTransform.ReverseTransform)
        # print("Transformed back:", pt2)
        # x = pt2.x()
        # y = pt2.y()

        x = point.x()
        y = point.y()
        self.statusbar.showMessage(f'x:{x}, y:{y},')
