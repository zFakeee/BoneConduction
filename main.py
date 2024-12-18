# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 744)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Triangular)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(0, 50, 231, 671))
        self.stackedWidget.setAutoFillBackground(False)
        self.stackedWidget.setStyleSheet("background-color: rgb(122, 122, 122);\n"
"background-color: rgb(90, 90, 90);\n"
"\n"
"\n"
"")
        self.stackedWidget.setObjectName("stackedWidget")
        self.stackedWidgetPage1 = QtWidgets.QWidget()
        self.stackedWidgetPage1.setObjectName("stackedWidgetPage1")
        self.pairingButton_4 = QtWidgets.QPushButton(self.stackedWidgetPage1)
        self.pairingButton_4.setGeometry(QtCore.QRect(10, 580, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(14)
        self.pairingButton_4.setFont(font)
        self.pairingButton_4.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(90, 90, 90);")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/pairing.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pairingButton_4.setIcon(icon)
        self.pairingButton_4.setIconSize(QtCore.QSize(60, 60))
        self.pairingButton_4.setObjectName("pairingButton_4")
        self.estimationButton_5 = QtWidgets.QPushButton(self.stackedWidgetPage1)
        self.estimationButton_5.setGeometry(QtCore.QRect(10, 500, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.estimationButton_5.setFont(font)
        self.estimationButton_5.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(90, 90, 90);")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/analyse.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.estimationButton_5.setIcon(icon1)
        self.estimationButton_5.setIconSize(QtCore.QSize(30, 30))
        self.estimationButton_5.setObjectName("estimationButton_5")
        self.ResamplingButton_3 = QtWidgets.QPushButton(self.stackedWidgetPage1)
        self.ResamplingButton_3.setGeometry(QtCore.QRect(10, 430, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.ResamplingButton_3.setFont(font)
        self.ResamplingButton_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"color: rgb(255, 255, 255);\n"
"background-color: rgb(90, 90, 90);")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/resampliing.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.ResamplingButton_3.setIcon(icon2)
        self.ResamplingButton_3.setIconSize(QtCore.QSize(30, 30))
        self.ResamplingButton_3.setObjectName("ResamplingButton_3")
        self.filteringButton_2 = QtWidgets.QPushButton(self.stackedWidgetPage1)
        self.filteringButton_2.setGeometry(QtCore.QRect(10, 270, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.filteringButton_2.setFont(font)
        self.filteringButton_2.setStyleSheet("background-color: rgb(90, 90, 90);\n"
"color: rgb(255, 255, 255);\n"
"")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/filter.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.filteringButton_2.setIcon(icon3)
        self.filteringButton_2.setIconSize(QtCore.QSize(30, 30))
        self.filteringButton_2.setObjectName("filteringButton_2")
        self.samplingButton = QtWidgets.QPushButton(self.stackedWidgetPage1)
        self.samplingButton.setGeometry(QtCore.QRect(10, 350, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.samplingButton.setFont(font)
        self.samplingButton.setStyleSheet("background-color: rgb(90, 90, 90);\n"
"color: rgb(255, 255, 255);\n"
"")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/song.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.samplingButton.setIcon(icon4)
        self.samplingButton.setIconSize(QtCore.QSize(40, 40))
        self.samplingButton.setObjectName("samplingButton")
        self.line_2 = QtWidgets.QFrame(self.stackedWidgetPage1)
        self.line_2.setGeometry(QtCore.QRect(0, -10, 1261, 20))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.label = QtWidgets.QLabel(self.stackedWidgetPage1)
        self.label.setGeometry(QtCore.QRect(30, 20, 151, 121))
        self.label.setStyleSheet("image: url(:/images/safe.png);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.gatherpushButton = QtWidgets.QPushButton(self.stackedWidgetPage1)
        self.gatherpushButton.setGeometry(QtCore.QRect(10, 190, 211, 51))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(12)
        self.gatherpushButton.setFont(font)
        self.gatherpushButton.setStyleSheet("color: rgb(255, 255, 255);\n"
"")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/images/data.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.gatherpushButton.setIcon(icon5)
        self.gatherpushButton.setIconSize(QtCore.QSize(30, 30))
        self.gatherpushButton.setObjectName("gatherpushButton")
        self.stackedWidget.addWidget(self.stackedWidgetPage1)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(230, 0, 1221, 721))
        self.widget.setStyleSheet("background-color: rgb(44, 44, 44);\n"
"")
        self.widget.setObjectName("widget")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(100, 160, 1001, 441))
        self.label_3.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"border-top-left-radius :45px;\n"
"border-bottom-left-radius :45px;\n"
"border-top-right-radius :45px;\n"
"border-bottom-right-radius :45px;")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.line = QtWidgets.QFrame(self.widget)
        self.line.setGeometry(QtCore.QRect(-12, -70, 31, 691))
        self.line.setStyleSheet("color: rgb(90, 90, 90);")
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.dataTextEdit = QtWidgets.QTextEdit(self.widget)
        self.dataTextEdit.setGeometry(QtCore.QRect(150, 160, 901, 441))
        self.dataTextEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"")
        self.dataTextEdit.setObjectName("dataTextEdit")
        self.pairtextEdit = QtWidgets.QTextEdit(self.widget)
        self.pairtextEdit.setGeometry(QtCore.QRect(450, 160, 301, 441))
        self.pairtextEdit.setStyleSheet("background-color: rgb(255, 255, 255);\n"
"font: 13pt \"宋体\";\n"
"")
        self.pairtextEdit.setObjectName("pairtextEdit")
        self.listView = QtWidgets.QListView(self.centralwidget)
        self.listView.setGeometry(QtCore.QRect(0, -10, 1401, 61))
        self.listView.setStyleSheet("background-color: rgb(90, 90, 90);\n"
"")
        self.listView.setObjectName("listView")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(-10, 0, 91, 51))
        self.label_2.setStyleSheet("image: url(:/images/system.png);")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(60, 10, 451, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color: rgb(0, 0, 0);")
        self.label_5.setObjectName("label_5")
        self.widget.raise_()
        self.stackedWidget.raise_()
        self.listView.raise_()
        self.label_2.raise_()
        self.label_5.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "骨传导配对系统"))
        self.pairingButton_4.setText(_translate("MainWindow", "设备配对"))
        self.estimationButton_5.setText(_translate("MainWindow", "信道估计"))
        self.ResamplingButton_3.setText(_translate("MainWindow", "重采样"))
        self.filteringButton_2.setText(_translate("MainWindow", "低通滤波"))
        self.samplingButton.setText(_translate("MainWindow", "声音转数字信号"))
        self.gatherpushButton.setText(_translate("MainWindow", "采集数据"))
        self.label_5.setText(_translate("MainWindow", "骨传导配对系统"))
import images_rc
