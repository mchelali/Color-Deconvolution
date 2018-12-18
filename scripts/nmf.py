# coding: utf8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
import cv2

class NMF_StainSeparation:
    def __init__(self, img):
        """
        :param  img: image d'une lame couleur RGB mais chaque canal est représenté dans une colone
                od: c'est l'image en dencité optique
                nmf_ : resultat de la decomposition
        :methode
                RGB2OD : transforme l'image en entrer en dencité optique
                lanchNMF : applique la decomposition NMF
        """
        # construction d'une matrice ou chaque colone represente un canal
        self.size = img.shape
        self.img = img.reshape(-1, 3)

        print self.img.shape
        self.od = None
        self.nmf_ = None # resultat de la transformation

    def RGB_2_OD(self):
        self.od = np.zeros([self.size[0]*self.size[1], self.size[2]])
        for i in range(self.size[0]*self.size[1]):
            for j in range(self.size[2]):
                if self.img[i, j] != 0:
                    self.od[i, j] = np.log10(self.img[i, j])

    def lanchNMF(self):
        model = NMF(n_components=3, init='random', random_state=0)
        self.nmf_ = model.fit_transform(self.img)

"""
Le main dans ce fichier sert a faire le test avant de l'implémenter dans l'interface


"""
if __name__ == "__main__":
    # charger une image
    img_path ="../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"
    img = plt.imread(img_path)

    img = img[:, :, 0:3]# Dans cette lligne je prend que les 3 canneaux et ne pas prendre le canal Alpha sur des images au format PNG
    l, c, d = img.shape
    #img = cv2.resize(img, (c/3, l/3))
    print img.shape

    nmf = NMF_StainSeparation(img) # Initialisation de la class n passant en parametre un image
    nmf.lanchNMF() # lancer la decomposition NMF

    resultat = nmf.nmf_.reshape((l, c, 3)) # le resultat obtenue n'est pas dans la meme taille en entrer
                                           # la chaque canal est representé dans une colone
                                           # je re-taille en taille initial pour pouvoire visualisé
    print resultat.shape

    plt.subplot(1, 4, 1)
    plt.imshow(img)

    plt.subplot(1, 4, 2)
    plt.imshow(resultat[:, :, 0], cmap="gray")

    plt.subplot(1, 4, 3)
    plt.imshow(resultat[:, :, 1], cmap="gray")

    plt.subplot(1, 4, 4)
    plt.imshow(resultat[:, :, 2], cmap="gray")

    plt.show()