from ventana_ui import *
from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import cv2 as cv

class Ventana(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        
        
        self.label.setText("")
        

        
        
    def setBackground(self, imgCV):
        height, width = imgCV.shape
        
        img = QtGui.QImage(imgCV, width, height, QtGui.QImage.Format_Grayscale8)
        img = QtGui.QPixmap.fromImage(img)
        
        self.label.setPixmap(img)
