# -*- coding: utf-8 -*-
from PySide import QtGui,QtCore
import sys
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from scripts import hsd as HSD
from scripts import OpticalDensity as OD
from scripts import ColorDeconvolution as Deconvolution
from scripts import ColorSpace as ColorSpace
import numpy as np


class Traitement_img(QtGui.QWidget):
    def __init__(self):
        super(Traitement_img, self).__init__()

        #---- La figure ou se place les images -----------
        self.figure = plt.figure()
        self.cluster=3

        self.canvas = FigureCanvas(self.figure)
        self.filename=None
        self.img=None


        #------ Les Boutons conecte avec sa fonction ------------
        self.label = QtGui.QLabel("Kmeans!")
        self.labell = QtGui.QPushButton("Choose number of cluster Kmeans!")
        self.labell.clicked.connect(self.openInputDialog)
        self.browse = QtGui.QPushButton('Parcourir')
        self.browse.clicked.connect(self.browse_on)
        self.hsi= QtGui.QPushButton('Chromaticite HSI')
        self.hsi.clicked.connect(self.calculate_HSI)
        self.hsd = QtGui.QPushButton('Chromaticite HSD')
        self.hsd.clicked.connect(self.calculate_HSD)
        #self.to_rotate = QtGui.QPushButton('Kmeans')
        #self.to_rotate.clicked.connect(self.openInputDialog)
        self.cluster = QtGui.QLineEdit('3')
        self.od = QtGui.QPushButton('Espace OD')
        self.od.clicked.connect(self.ODSpace)
        self.deconvolution = QtGui.QPushButton('Color deconvolution')
        self.deconvolution.clicked.connect(self.colorDeconvolution)
        self.reconstruction = QtGui.QPushButton('Image HSI')
        self.reconstruction.clicked.connect(self.reconstructionToRGB)
        self.reconstructionHSD = QtGui.QPushButton('Image HSD')
        self.reconstructionHSD.clicked.connect(self.reconstructionToRGBHSD)

        self.normeL = QtGui.QPushButton('Espace de representation normeL1')
        self.normeL.clicked.connect(self.normeL1)

        self.DigitalComplet = QtGui.QPushButton('Espace de representation passageAuCubeDigitalComplet')
        self.DigitalComplet.clicked.connect(self.CubeDigitalComplet)

        self.norme = QtGui.QPushButton('Espace de representation norme max-min')
        self.norme.clicked.connect(self.maxmin)
        #---rotation layout------
        rot = QtGui.QHBoxLayout()
        #rot.addWidget(self.label)
        #rot.addWidget(self.cluster)
        rot.addWidget(self.labell)


        #----Bouton Layout_____
        lbo = QtGui.QVBoxLayout()
        lbo.addWidget(self.browse)
        lbo.addWidget(self.hsi)
        lbo.addWidget(self.hsd)
        lbo.addLayout(rot)
        lbo.addWidget(self.reconstruction)
        lbo.addWidget(self.reconstructionHSD)
        lbo.addWidget(self.od)
        lbo.addWidget(self.deconvolution)
        lbo.addWidget(self.normeL)
        lbo.addWidget(self.DigitalComplet)
        lbo.addWidget(self.norme)

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
        print self.filename[65:]
        self.img = plt.imread(self.filename)
        self.img = self.img[:, :, 0:3].astype(np.double)

        #self.filename=filename
        #print self.filename
        if self.filename != '':
            # self.print_img(filename)
            self.mat = plt.imread(self.filename)
            # plt.imshow(self.mat)
            # self.figure.add_subplot(211)
            plt.imshow(self.mat)
            self.canvas.draw()



    def calculate_HSI(self):
        # hsi
        h = HSD.HSD(self.img, self.filename, "intensity")
        h.chromaticite()
        h.Kmeans2(self.cluster)

    def calculate_HSD(self):

        od=OD.rgb_2_od2(self.img)
        h = HSD.HSD(od, self.filename, "Density")
        h.chromaticite()
        #Z = h.Kmeans(cluster)
        #h.plotHS(Z)
        h.Kmeans2(self.cluster)

    def reconstructionToRGB(self):
        h = HSD.HSD(self.img, self.filename, "intensity")
        h.chromaticite()
        h.calcule_HSI()
        h.recontructionToRGB()
        h.binarisation()

    def ODSpace(self):
        od=OD.rgb_2_od(self.img)


    def colorDeconvolution(self):
        img2 = self.img[:, :, 0:3].astype(np.uint8).copy()
        deconvolution = Deconvolution.ColorDeconvolution(img2,self.filename)
        deconvolution.RGB_2_OD()
        deconvolution.separateStain()
        deconvolution.showStains()

    def reconstructionToRGBHSD(self):
        img2 = self.img[:, :, 0:3].astype(np.uint8).copy()
        dec = Deconvolution.ColorDeconvolution(img2, self.filename)
        od=dec.RGB_2_OD()
        h = HSD.HSD(od, self.filename, "Density")
        h.chromaticite()
        h.calcule_HSI()
        h.recontructionToRGB()
        h.binarisation()
    def normeL1(self):

        path1 = self.filename[:49] + "Resultat/" + self.filename[65:] + "/HSD/HSI_Density.tif"
        path2 = self.filename[:49] + "Resultat/" + self.filename[65:] + "/HSD/HSI_Saturation.tif"
        path3 = self.filename[:49] + "Resultat/" + self.filename[65:] + "/HSD/HSI_Teinte.tif"


        img1 = plt.imread(path1)
        img2 = plt.imread(path2)
        img3 = plt.imread(path3)
        img = np.zeros([img1.shape[0], img1.shape[1], 3])
        img[:, :, 0] = img3[:, :]
        img[:, :, 1] = img2[:, :]
        img[:, :, 2] = img1[:, :]

        color = ColorSpace.ColorSpace(img)
        color.normalise()

        img_hsv = color.HSV()
        img_hsl = color.HSL()
        color.transformationConiqueHSV()
        color.transformationConiqueHSL()
        color.luminanceSaturationTeinteL1()

    def CubeDigitalComplet(self):

        path1 = self.filename[:49] + "Resultat/" + self.filename[65:] + "/HSD/HSI_Density.tif"
        path2 = self.filename[:49] + "Resultat/" + self.filename[65:] + "/HSD/HSI_Saturation.tif"
        path3 = self.filename[:49] + "Resultat/" + self.filename[65:] + "/HSD/HSI_Teinte.tif"

        img1 = plt.imread(path1)
        img2 = plt.imread(path2)
        img3 = plt.imread(path3)
        img = np.zeros([img1.shape[0], img1.shape[1], 3])
        img[:, :, 0] = img3[:, :]
        img[:, :, 1] = img2[:, :]
        img[:, :, 2] = img1[:, :]

        color = ColorSpace.ColorSpace(img)
        color.normalise()

        img_hsv = color.HSV()
        img_hsl = color.HSL()
        color.transformationConiqueHSV()
        color.transformationConiqueHSL()
        color.passageAuCubeDigitalComplet()

    def maxmin(self):


        path1 = self.filename[:49]+"Resultat/"+self.filename[65:] + "/HSD/HSI_Density.tif"
        path2 = self.filename[:49]+"Resultat/"+self.filename[65:] + "/HSD/HSI_Saturation.tif"
        path3 = self.filename[:49]+"Resultat/"+self.filename[65:] + "/HSD/HSI_Teinte.tif"

        img1 = plt.imread(path1)
        img2 = plt.imread(path2)
        img3 = plt.imread(path3)
        img = np.zeros([img1.shape[0], img1.shape[1], 3])
        img[:, :, 0] = img3[:, :]
        img[:, :, 1] = img2[:, :]
        img[:, :, 2] = img1[:, :]
        color = ColorSpace.ColorSpace(self.img)
        color.normalise()

        img_hsv = color.HSV()
        img_hsl = color.HSL()
        color.transformationConiqueHSV()
        color.transformationConiqueHSL()
        color.normMaxMin()

    def openInputDialog(self):
        """
        Opens the text version of the input dialog
        """
        text, result = QtGui.QInputDialog.getText(self, "I'm a text Input Dialog!",
                                                  "How many cluster you want please max 4?")
        if result:
            print "Number of cluster you choose is %s!" % text
            self.cluster=int(text)



if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    ex = Traitement_img()
    sys.exit(app.exec_())
