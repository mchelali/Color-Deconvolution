import matplotlib.pyplot as plt
import numpy as np
import os

class PCA:
    def __init__(self, path=""):
        # constructeur de la class
        self.path_DB = path #chemin ver la DataSet
        self.__cellsNumber = None # nombre de cellules
        self.__cells = None #Matrice de cellules
        self.__callShape = None #taille of the images in the DataSet

    def startPCA(self):
        self.__readDataSet__()
        self.__avreageCell__()

    #-------------- Getter and Setters ------------------------------------

    def setPath(self, path=""):
        if path == "":
            print("Please give a path to your DataSet")
        else:
            self.path_DB = path

    def getPath(self):
        return self.path_DB

    def getCellNumber(self):
        return self.__cellsNumber

    #----------------------------------------------------------------------

    def __readDataSet__(self):
        if self.path_DB == "":
            print("give a path to the DataSet first")
        else:
            r = []
            g = []
            b = []
            for dirname, dirnames, filenames in os.walk(self.path_DB):
                for subdirname in dirnames:
                    subject_path = os.path.join(dirname, subdirname)
                    for filename in os.listdir(subject_path):
                        if (filename.endswith('tiff')) :
                            # print os.path.join(subject_path, filename)
                            im = plt.imread(os.path.join(subject_path, filename))
                            r.append(im[:, :, 0].ravel())
                            g.append(im[:, :, 1].ravel())
                            b.append(im[:, :, 2].ravel())
            self.__cellsNumber = len(r)
            self.__cells = np.array([r, g, b]).transpose()

    def __avreageCell__(self):
        # calculer la cellule maoyenne de tous les cellules
        self.avreageCell = np.zeros([len(self.__cells[0, :]), 3])# init d'un vecteur 0
        for i in range(self.__cells.shape[0]):
            self.avreageCell[i] = np.round((sum(self.__cells[i, :]) / self.__cellsNumber), 3)