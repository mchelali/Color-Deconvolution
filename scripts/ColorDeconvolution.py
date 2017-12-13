#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np

class ColorDeconvolution:
    def __init__(self, img):
        self.img_0 = img
        self.stains = None
        self.od = None

    def setImage(self, img):
        if img != None:
            self.img_0 = img

    def RGB_2_OD(self):
        [l, c, d] = self.img_0.shape
        self.od = np.zeros([l, c, d])
        for i in range(l):
            for j in range(c):
                for k in range(d):
                    if self.img_0[i,j,k] != 0 :
                        self.od[i,j,k] = np.log(self.img_0[i, j, k])
        return self.od

    def norm(self, vector):
        n = 0
        for i in vector:
            n = n + np.square(i)
        return np.sqrt(n)

    def separateStain(self):

        # set of standard values for stain vectors (from python scikit)
        # He = [0.65; 0.70; 0.29];
        # Eo = [0.07; 0.99; 0.11];
        # DAB = [0.27; 0.57; 0.78];

        # alternative set of standard values (HDAB from Fiji)
        He = np.array([0.6500286, 0.704031, 0.2860126])  # Hematoxylin
        Eo = np.array([0.07, 0.99, 0.11])  # Eosine
        DAB = np.array([0.26814753, 0.57031375, 0.77642715])  # DAB
        Res = np.array([0.7110272, 0.42318153, 0.5615672])  # residual

        # combine stain vectors to deconvolution matrix
        HDABtoRGB = np.array([He / self.norm(He), Eo / self.norm(Eo), DAB / self.norm(DAB)])
        RGBtoHDAB = np.linalg.inv(HDABtoRGB)


        [l,c,d] = self.img_0.shape
        self.stains = np.zeros([l, c, d])
        for i in range(l):
            for j in range(c):
                a = np.dot(self.od[i, j], RGBtoHDAB)
                b=self.od[i,j]
                self.stains[i, j, 0] = a[0]
                self.stains[i, j, 1] = a[1]
                self.stains[i, j, 2] = a[2]

    def showStains(self):

        plt.subplot(1, 4, 1)
        plt.title("original")
        plt.imshow(self.img_0)

        plt.subplot(1, 4, 2)
        plt.title('Hematoxylin')
        plt.imshow(self.stains[:, :, 0], cmap="gray")

        plt.subplot(1, 4, 3)
        plt.title('Eosine')
        plt.imshow(self.stains[:, :, 1], cmap="gray")

        plt.subplot(1, 4, 4)
        plt.title('DAB')
        plt.imshow(self.stains[:, :, 2], cmap="gray")

        plt.show()

if __name__=="__main__":

    # reading the image
    #path="Tumor_CD31_LoRes.png"
    path="../figure9.jpg"
    img = plt.imread(path)
    img = img[:, :, 0:3]


    # deconvolution de couleur
    satin = ColorDeconvolution(img)
    satin.RGB_2_OD()
    satin.separateStain()
    satin.showStains()