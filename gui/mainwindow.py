#coding: utf-8

import sys
from PySide import QtGui
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from scripts import ColorDeconvolution as CD

class MainWindow(QtGui.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.mat=None # cette variable contient l'image charger
        self.initUI()

    def initUI(self):
        # ---- La figure ou se place les images -----------
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.setCentralWidget(self.canvas)
        # ------ Les Boutons conecte avec sa fonction ------------
        load = QtGui.QAction(QtGui.QIcon(""), "Load Image", self) # Ce boutton permet de charger une image
        load.triggered.connect(self.browse_on)
        exitAction = QtGui.QAction(QtGui.QIcon(""), 'Exit', self) # Ce boutton permet de fermer l'application
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)

        # toolbar
        self.toolbar = self.addToolBar("ToolBar")
        self.toolbar.addAction(load)
        self.toolbar.addAction(exitAction)

        self.setWindowTitle('Stain separation')

        self.show()

    def browse_on(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(None, 'Open file')[0])
        print filename
        if filename != '':
            # self.print_img(filename)
            self.mat = plt.imread(filename)
            self.print_img(self.mat)

    def print_img(self, matrice):
        if len(matrice.shape) == 3:
            plt.imshow(matrice)
        else:
            plt.imshow(matrice, cmap=cm.Greys_r)
        plt.axis('off')
        self.canvas.draw()

def main():
    app = QtGui.QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()