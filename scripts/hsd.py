#coding:  utf-8
import numpy as np
import matplotlib.pyplot as plt
import OpticalDensity as od

class HSD:
    def __init__(self, img):
        """
        :param img: image en entrer pour changer d'espace de couleur
            chroma : matrice de chromaticité. de dimension
                    [
                     l: nbr de ligne ,
                     c: nbr de colone,
                     d: dimension 2 qui represente les coordonée chromatique
                    ]
        :fonctions :
            - rgb_2_gray : transfomé en gris
            - chromaticite : calculer les coordonnées chromatiques
            - calcule_HSI : calculer le nouvel espace Hue Saturation Intensity
        """
        self.img_0 = img
        self.img_hsi = None
        self.chroma = None

    def RGB_2_GRAY(self):
        [l, c, d] = self.img_0.shape
        gray = np.zeros([l, c])
        for i in range(l):
            for j in range(c):
                gray[i, j] = sum(self.img_0[i, j])/3
        return gray

    def chromaticite(self):
        [l, c, d] = self.img_0.shape
        gray = self.RGB_2_GRAY()
        self.chroma = np.zeros([l, c, 2])
        for i in range(l):
            for j in range(c):
                self.chroma[i, j ,0] = (self.img_0[i, j, 0] / gray[i, j]) - 1
                self.chroma[i, j, 1] = (self.img_0[i, j, 1] - self.img_0[i, j, 2]) / (gray[i, j] * np.sqrt(3))

    def calcule_HSI(self):
        [l, c, d] = self.img_0.shape
        gray = self.RGB_2_GRAY()
        self.img_hsi = np.zeros([l, c, 3])
        self.img_hsi[:, :, 2] = gray
        for i in range(l):
            for j in range(c):
                self.img_hsi[i, j, 0] = self.getHue2(self.chroma[i, j, 0], self.chroma[i, j ,1])
                self.img_hsi[i, j, 1] = self.getSaturation2(self.chroma[i, j ,0], self.chroma[i, j ,1])

    def getSaturation2(self, cx, cy):
        return np.sqrt(np.square(cx)+np.square(cy))

    def getHue2(self, cx, cy):
        return np.arctan(cy/cx)

    def plotHSD(self):
        plt.subplot(1,3,1)
        plt.imshow(self.img_hsi[:,:,0], cmap="gray")

        plt.subplot(1,3,2)
        plt.imshow(self.img_hsi[:, :, 1], cmap="gray")

        plt.subplot(1, 3, 3)
        plt.imshow(self.img_hsi[:, :, 2], cmap="gray")
        plt.show()

if __name__ == "__main__":
    # reading the image
    # path="Tumor_CD31_LoRes.png"
    path = "../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"
    img = plt.imread(path)
    img = img[:, :, 0:3].astype(np.double)

    # hsi
    h = HSD(img)
    h.chromaticite()
    h.calcule_HSI()
    h.plotHSD()

    #calculer la OD
    OD = od.rgb_2_od(img)

    # hsd
    #h1 = HSD(OD)
    #h1.chromaticite()
    #h1.calcule_HSI()
    #h1.plotHSD()


