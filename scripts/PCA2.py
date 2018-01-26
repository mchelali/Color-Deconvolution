import matplotlib.pyplot as plt
import numpy as np
import OpticalDensity as OD

from sklearn.decomposition import FastICA, PCA


class PC:
    def __init__(self,img0):
        self.img=img0
        self.reduction=None
        self.acp=None
        self.od=OD.rgb_2_od2(self.img)
        self.S=None
        self.l=self.img.shape[0]
        self.c=self.img.shape[1]

    def pca(self):
        pca = PCA(n_components=3)

        r = self.od[:, :, 0]
        g = self.od[:, :, 1]
        b = self.od[:, :, 2]

        r = r.ravel()
        g = g.ravel()
        b = b.ravel()
        img_new=np.zeros((self.img.shape[0]*self.img.shape[1],3))
        img_new[:, 0] = r
        img_new[:, 1] = g
        img_new[:, 2] = b

        self.acp=pca.fit_transform(img_new)
        #print pca.singular_values_
        #print self.reduction
        #print self.reduction.shape

    def matriceReduite(self):

        r = self.od[:, :, 0]
        g = self.od[:, :, 1]
        b = self.od[:, :, 2]

        r = r.ravel()
        g = g.ravel()
        b = b.ravel()
        img_new = np.zeros((self.img.shape[0] * self.img.shape[1], 3))
        img_new[:, 0] = r
        img_new[:, 1] = g
        img_new[:, 2] = b

        self.reduction = np.dot(img_new.T,self.acp)
        print self.reduction.shape

    def fastICA(self):
        # Compute ICA
        ica = FastICA(n_components=3,algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None)
        self.S = ica.fit_transform(self.reduction)  # Reconstruct signals
        A_ = ica.mixing_  # Get estimated mixing matrix
        for i in range(self.S.shape[0]):
            if (self.S[i, 0] < 0):
                self.S[i, 0] = 0
            if (self.S[i, 1] < 0):
                self.S[i, 1] = 150
            if (self.S[i, 2] < 0):
                self.S[i, 2] = 150


        for i in range(self.reduction.shape[0]):
            if (self.reduction[i, 0] < 0):
                self.reduction[i, 0] = 255
            if (self.reduction[i, 1] < 0):
                self.reduction[i, 1] = 0
            if (self.reduction[i, 2] < 0):
                self.reduction[i, 2] = 0


        self.S = np.reshape(self.S, ((self.l, self.c, 3)))
        self.reduction = np.reshape(self.reduction, ((self.l, self.c, 3)))
        print self.S.shape
        print self.S
        print self.reduction
        print self.img




        plt.subplot(1, 2, 1)
        plt.title("ICA")
        plt.imshow(self.S)
        plt.subplot(1, 2, 2)
        plt.title("PCA")
        plt.imshow(self.reduction)
        plt.show()


if __name__ == '__main__':
    path="../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"
    img=plt.imread(path)
    p=PC(img)
    p.pca()
    p.matriceReduite()

    p.fastICA()