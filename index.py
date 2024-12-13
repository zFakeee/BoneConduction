from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.uic import loadUiType
import sys

ui, _ = loadUiType('main.ui')

class Mainapp(QMainWindow,ui):
     #定义构造方法
     def __int__(self):
         QMainWindow.__init__(self)
         self.setupUi(self)

def main():
    app = QApplication([])
    window = Mainapp()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
