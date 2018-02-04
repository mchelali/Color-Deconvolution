# coding: utf8
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import NMF
import cv2

class NMF_StainSeparation:
    def __init__(self, img):
        """
        :param img: image d'une lame couleur RGB
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


if __name__ == "__main__":
    img_path ="../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"
    img_path = "../DataSet_Lomenie/arn2.tif"
    #img_path = "../DataSet_Lomenie/Lung_4stain.tif"
    #img_path = "../Stain/Lung_4stain02.tif"
    #img_path = "../he.png"

    img = plt.imread(img_path)
    img = img[:, :, 0:3]
    l, c, d = img.shape
    #img = cv2.resize(img, (c/3, l/3))
    print img.shape

    nmf = NMF_StainSeparation(img)
    nmf.lanchNMF()

    resultat = nmf.nmf_.reshape((l, c, 3))

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