# -*- coding: utf-8 -*-
from PySide import QtGui,QtCore
import sys
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from scripts import hsd as HSD
import numpy as np


class Traitement_img(QtGui.QWidget):
    def __init__(self):
        super(Traitement_img, self).__init__()

        #---- La figure ou se place les images -----------
        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)
        self.filename=None


        #------ Les Boutons conecte avec sa fonction ------------
        self.browse = QtGui.QPushButton('Parcourir')
        self.browse.clicked.connect(self.browse_on)
        self.hsi= QtGui.QPushButton('HSI')
        self.hsi.clicked.connect(self.calculate_HSI)
        self.to_rotate = QtGui.QPushButton('Kmeans')
        #self.to_rotate.clicked.connect(self.rotation)
        self.cluster = QtGui.QLineEdit('rentrez le nombre de cluster')

        #---rotation layout------
        rot = QtGui.QHBoxLayout()
        rot.addWidget(self.cluster)
        rot.addWidget(self.to_rotate)

        #----Bouton Layout_____
        lbo = QtGui.QVBoxLayout()
        lbo.addWidget(self.browse)
        lbo.addWidget(self.hsi)
        lbo.addLayout(rot)

        #------- Le Layout Principal -----
        layout = QtGui.QGridLayout()
        layout.addWidget(self.canvas, 0, 0, 5, 5)

        layout.addLayout(lbo, 0, 5)

        #---------------------------
        self.setLayout(layout)


        self.setWindowTitle(u'Traitement d\'image')
        self.show()

    def browse_on(self):
        self.filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open file')[0])
        #self.filename=filename
        print self.filename
        if self.filename != '':
            # self.print_img(filename)
            self.mat = plt.imread(self.filename)
            plt.imshow(self.mat)
            #self.figure.add_subplot(211)
            #plt.imshow(self.mat)
            self.canvas.draw()

    def calculate_HSI(self):
        cluster = int(self.cluster.text())
        self.canvas.figure.clf()
        img = plt.imread(self.filename)
        img = img[:, :, 0:3].astype(np.double)

        # hsi
        h = HSD.HSD(img)
        h.chromaticite()
        Z=h.Kmeans(cluster)
        h.plotHS(Z)


if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    ex = Traitement_img()
    sys.exit(app.exec_())