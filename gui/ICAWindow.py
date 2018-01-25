#coding: utf-8

import sys
from PySide import QtGui, QtCore
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from scripts import pca as Stain
import numpy as np

class ICAWindow(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.img = []
        self .initUi()

    def initUi(self):

        # ---- La figure ou se place les images -----------
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)

        # ---- declaration des bouttons -----
        self.save = QtGui.QPushButton(u"Enregistré les résultats")
        self.save.clicked.connect(self.save_result)
        self.close_ = QtGui.QPushButton(u"Fermer")
        self.close_.clicked.connect(self.close)

        # ----- Layout des bouttons ------
        la = QtGui.QHBoxLayout()
        la.addWidget(self.save)
        la.addWidget(self.close_)

        # -- principal Layout ---
        layout = QtGui.QGridLayout()
        layout.addWidget(self.canvas, 0, 0, 5, 10)
        layout.addLayout(la, 5, 0)

        self.setLayout(layout)


        self.setWindowTitle('Resultat')

    def addImage(self, image):
        self.img.append(image)

    def show_(self):
        print len(self.img)
        pos = 0
        for i in range(len(self.img)):
            plt.subplot(len(self.img), 3, pos+1)
            plt.imshow(self.img[i][:, :, 0], cmap="gray")
            plt.subplot(len(self.img), 3, pos+2)
            plt.imshow(self.img[i][:, :, 1], cmap="gray")
            plt.subplot(len(self.img), 3, pos+3)
            plt.imshow(self.img[i][:, :, 2], cmap="gray")
            pos = pos+3

        self.canvas.draw()

    def save_result(self):
        path = str(QtGui.QFileDialog.getExistingDirectory(self, "Select Directory"))
        for i in range(len(self.img)):
            cv2.imwrite(path + "/pca1_"+str(i)+".png", self.img[i][:, :, 0])
            cv2.imwrite(path + "/pca2_"+str(i)+".png", self.img[i][:, :, 1])
            cv2.imwrite(path + "/pca3_"+str(i)+".png", self.img[i][:, :, 2])