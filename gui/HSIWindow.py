# -*- coding: utf-8 -*-
from PySide import QtGui,QtCore
import sys
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from scripts import hsd as HSD
from scripts import OpticalDensity as OD
from scripts import ColorDeconvolution as Deconvolution
import numpy as np


class Traitement_img(QtGui.QWidget):
    def __init__(self):
        super(Traitement_img, self).__init__()

        #---- La figure ou se place les images -----------
        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)
        self.filename=None
        self.img=None


        #------ Les Boutons conecte avec sa fonction ------------
        self.label = QtGui.QLabel("Kmeans!")
        self.browse = QtGui.QPushButton('Parcourir')
        self.browse.clicked.connect(self.browse_on)
        self.hsi= QtGui.QPushButton('Chromaticite HSI')
        self.hsi.clicked.connect(self.calculate_HSI)
        #self.to_rotate = QtGui.QPushButton('Kmeans')
        #self.to_rotate.clicked.connect(self.rotation)
        self.cluster = QtGui.QLineEdit('Nombre de cluster')
        self.od = QtGui.QPushButton('Espace OD')
        self.od.clicked.connect(self.ODSpace)
        self.deconvolution = QtGui.QPushButton('Color deconvolution')
        self.deconvolution.clicked.connect(self.colorDeconvolution)
        self.reconstruction = QtGui.QPushButton('Image HSI')
        self.reconstruction.clicked.connect(self.reconstructionToRGB)
        self.reconstructionHSD = QtGui.QPushButton('Image HSD')
        self.reconstructionHSD.clicked.connect(self.reconstructionToRGBHSD)

        #---rotation layout------
        rot = QtGui.QHBoxLayout()
        rot.addWidget(self.label)
        rot.addWidget(self.cluster)
        #rot.addWidget(self.to_rotate)

        #----Bouton Layout_____
        lbo = QtGui.QVBoxLayout()
        lbo.addWidget(self.browse)
        lbo.addWidget(self.hsi)
        lbo.addLayout(rot)
        lbo.addWidget(self.reconstruction)
        lbo.addWidget(self.reconstructionHSD)
        lbo.addWidget(self.od)
        lbo.addWidget(self.deconvolution)

        #------- Le Layout Principal -----
        layout = QtGui.QGridLayout()
        layout.addWidget(self.canvas, 0, 0, 5, 5)

        layout.addLayout(lbo, 0, 5)

        #---------------------------
        self.setLayout(layout)


        self.setWindowTitle(u'Traitement d\'image')
        self.show()

    def browse_on(self):
        self.canvas.figure.clf()
        self.filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open file')[0])
        self.img = plt.imread(self.filename)
        #self.img = self.img[:, :, 0:3].astype(np.double)

        #self.filename=filename
        print self.filename
        if self.filename != '':
            # self.print_img(filename)
            self.mat = plt.imread(self.filename)
            # plt.imshow(self.mat)
            # self.figure.add_subplot(211)
            plt.imshow(self.mat)
            self.canvas.draw()



    def calculate_HSI(self):
        cluster = int(self.cluster.text())
        #self.canvas.figure.clf()

        # hsi
        h = HSD.HSD(self.img)
        h.chromaticite()
        Z = h.Kmeans(cluster)
        h.plotHS(Z)


    def reconstructionToRGB(self):
        h = HSD.HSD(self.img)
        h.chromaticite()
        h.calcule_HSI()
        h.recontructionToRGB()

    def ODSpace(self):
        od=OD.rgb_2_od(self.img)



    def colorDeconvolution(self):
        deconvolution = Deconvolution.ColorDeconvolution(self.img)
        deconvolution.RGB_2_OD()
        deconvolution.separateStain()
        deconvolution.showStains()

    def reconstructionToRGBHSD(self):
        dec=Deconvolution.ColorDeconvolution(self.img)
        od=dec.RGB_2_OD()
        h = HSD.HSD(od)
        h.chromaticite()
        h.calcule_HSI()
        h.recontructionToRGB()


if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    ex = Traitement_img()
    sys.exit(app.exec_())