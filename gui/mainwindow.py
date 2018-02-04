#coding: utf-8

import sys
from PySide import QtGui, QtCore
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from scripts import pca as Stain
import ICAWindow as methodes
import numpy as np

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.image = None # Input Image

        self.initUI()

    def initUI(self):
        # ---- La figure ou se place les images -----------
        self.printer = QtGui.QPrinter()
        self.scaleFactor = 0.0

        self.imageLabel = QtGui.QLabel()
        self.imageLabel.setBackgroundRole(QtGui.QPalette.Base)
        self.imageLabel.setSizePolicy(QtGui.QSizePolicy.Ignored,
                                      QtGui.QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QtGui.QScrollArea()
        self.scrollArea.setBackgroundRole(QtGui.QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.setCentralWidget(self.scrollArea)
        self.createActions()
        self.createMenus()



        self.resize(500, 400)

        self.setWindowTitle('Stain separation')

        self.show()

    def browse_on(self):
        fileName, _ = QtGui.QFileDialog.getOpenFileName(self, "Open File",
                                                        QtCore.QDir.currentPath())
        if fileName:
            """
            Lire l'image si le chemin n'est pas vide

            Aussi ici je fait la gestion du typr d'image pour pouvoir le convertir du type numpy.ndarray a QImage
            """
            self.image = plt.imread(fileName)
            self.print_img(self.image)
            self.l, self.c, self.d = self.image.shape
            self.stain = Stain.ICA_StainSeparation(self.image.astype(np.float))



    def print_img(self, image):
        imageQt = None
        if len(image.shape) == 2:
            imageQt = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_Indexed8)
        elif len(image.shape) == 3:
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[2] == 3:
                print "shape = 3"
                imageQt = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[2] * image.shape[1],
                                       QtGui.QImage.Format_RGB888)

                """elif image.shape[2] == 4:
                    print "shape = 4"
                    print image[0,0]
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    imageQt = QtGui.QImage(image.data, image.shape[1], image.shape[0], image.shape[2] * image.shape[1],
                                           QtGui.QImage.Format_RGB888)"""
            else:
                imageQt = QtGui.QImage()
        if imageQt.isNull():
            QtGui.QMessageBox.information(self, "Image Viewer", "Cannot load .")
            return
        self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(imageQt))
        self.scaleFactor = 1.0

        self.printAct.setEnabled(True)
        self.fitToWindowAct.setEnabled(True)
        self.updateActions()

        if not self.fitToWindowAct.isChecked():
            self.imageLabel.adjustSize()

    def print_(self):
        dialog = QtGui.QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QtGui.QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def about(self):
        QtGui.QMessageBox.about(self, "Stain separation",
                                """
                                Cette application a ete cree dans le context du projet de fin
                                d'etude de master 2 en Image et Pluremedia
                                """)

    def createActions(self):
        """
        Creation des actions qu'on peut faire ex: pour ouverir un fichier, zoom ....
        """
        self.openAct = QtGui.QAction("&Open...", self, shortcut="Ctrl+O",
                                     triggered=self.browse_on)

        self.printAct = QtGui.QAction("&Print...", self, shortcut="Ctrl+P",
                                      enabled=False, triggered=self.print_)

        self.exitAct = QtGui.QAction("E&xit", self, shortcut="Ctrl+Q",
                                     triggered=self.close)

        self.zoomInAct = QtGui.QAction("Zoom &In (25%)", self,
                                       shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)

        self.zoomOutAct = QtGui.QAction("Zoom &Out (25%)", self,
                                        shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)

        self.normalSizeAct = QtGui.QAction("&Normal Size", self,
                                           shortcut="Ctrl+S", enabled=False, triggered=self.normalSize)

        self.fitToWindowAct = QtGui.QAction("&Fit to Window", self,
                                            enabled=False, checkable=True, shortcut="Ctrl+F",
                                            triggered=self.fitToWindow)

        self.aboutAct = QtGui.QAction("&About", self, triggered=self.about)

        self.aboutQtAct = QtGui.QAction("About &Qt", self,
                                        triggered=QtGui.qApp.aboutQt)
        self.pca = QtGui.QAction("pca methode", self,
                                        triggered=self.pca_methode)
        self.ica = QtGui.QAction("ica method", self,
                                        triggered=self.ica_methode)

        self.binPca = QtGui.QAction("binarize pca", self,
                                        triggered=self.pca_binarization)
        self.binICA = QtGui.QAction("binarize ica", self,
                                        triggered=self.ica_binarization)

        self.ruifrok = QtGui.QAction("Methode Ruifrok", self,
                                        triggered=self.ruifrokStainSeparation)

        self.nmf = QtGui.QAction("NMF", self,
                                        triggered=self.nmf_methode)

    def createMenus(self):
        """
        Organisation des actions dans des menus
        """
        self.fileMenu = QtGui.QMenu("&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.meth = QtGui.QMenu("&Methode", self)
        self.meth.addAction(self.ruifrok)
        self.meth.addSeparator()
        self.meth.addAction(self.nmf)
        self.meth.addSeparator()
        self.meth.addAction(self.pca)
        self.meth.addAction(self.binPca)
        self.meth.addSeparator()
        self.meth.addAction(self.ica)
        self.meth.addAction(self.binICA)

        self.viewMenu = QtGui.QMenu("&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.helpMenu = QtGui.QMenu("&Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.meth)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.helpMenu)


    def updateActions(self):
        self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
        self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.333)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))


    def ruifrokStainSeparation(self):
        from scripts import ColorDeconvolution as cd
        # deconvolution de couleur
        satin = cd.ColorDeconvolution(self.image, "")
        satin.RGB_2_OD()
        image = satin.separateStain()
        self.cd_view = methodes.ICAWindow("ruifrock_")
        self.cd_view.addImage(image)
        self.cd_view.show_()
        self.cd_view.show()

    def pca_methode(self):
        print "pca work"
        self.stain.RGB_2_OD()
        self.stain.lanchePCA()
        self.img_pca = self.stain.normalisation2(self.stain.pca_).reshape((self.l, self.c, 3))
        self.pca_view = methodes.ICAWindow("pca_")
        self.pca_view.addImage(self.img_pca)
        self.pca_view.show_()
        self.pca_view.show()

    def ica_methode(self):
        print "ica work"
        self.stain.lanchICA()
        self.img_ica = self.stain.normalisation2(self.stain.ica_).reshape((self.l, self.c, 3))
        self.ica_view = methodes.ICAWindow("ica_")
        self.ica_view.addImage(self.img_ica)
        self.ica_view.show_()
        self.ica_view.show()

    def pca_binarization(self):
        print "binarization pca"
        pca1 = np.uint8(self.img_pca[:, :, 0])
        pca2 = np.uint8(self.img_pca[:, :, 1])
        pca3 = np.uint8(self.img_pca[:, :, 2])

        bin_pca = np.zeros([self.l, self.c, self.d])
        thres1, bin_pca[:, :, 0] = cv2.threshold(pca1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thres2, bin_pca[:, :, 1] = cv2.threshold(pca2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thres3, bin_pca[:, :, 2] = cv2.threshold(pca3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.pca_view.addImage(bin_pca)
        self.pca_view.show_()
        self.pca_view.show()

    def ica_binarization(self):
        print "binarization ica"
        ica1 = np.uint8(self.img_ica[:, :, 0])
        ica2 = np.uint8(self.img_ica[:, :, 1])
        ica3 = np.uint8(self.img_ica[:, :, 2])

        bin_ica = np.zeros([self.l, self.c, self.d])
        thres1, bin_ica[:, :, 0] = cv2.threshold(ica1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thres2, bin_ica[:, :, 1] = cv2.threshold(ica2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thres3, bin_ica[:, :, 2] = cv2.threshold(ica3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.ica_view.addImage(bin_ica)
        self.ica_view.show_()
        self.ica_view.show()

    def nmf_methode(self):
        from scripts import nmf
        nmf = nmf.NMF_StainSeparation(self.image)
        nmf.lanchNMF()
        resultat = nmf.nmf_.reshape((self.l, self.c, 3))
        self.nmf_view = methodes.ICAWindow("nmf_")
        self.nmf_view.addImage(resultat)
        self.nmf_view.show_()
        self.nmf_view.show()


def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()