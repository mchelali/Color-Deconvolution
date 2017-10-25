import matplotlib.pyplot as plt
import numpy as np
import os

class PCA:
    def __init__(self, img):
        # constructeur de la class
        self.img_0 = img
        self.od = None

    def getOD(self):
        return self.od

    def RGB_2_OD(self):
        [l, c, d] = self.img_0.shape
        self.od = np.zeros([l, c, d])
        for i in range(l):
            for j in range(c):
                for k in range(d):
                    if self.img_0[i, j, k] != 0:
                        self.od[i, j, k] = np.log(self.img_0[i, j, k])

    def getComponents(self, image_2d, numStain=100):
        cov_mat = image_2d - np.average(image_2d)
        eig_val, eig_vec = np.linalg.eigh(np.cov(cov_mat))  # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
        p = len(eig_vec)
        print p
        print (eig_vec.shape)
        idx = np.argsort(eig_val)
        idx = idx[::-1]
        eig_vec = eig_vec[:, idx]
        eig_val = eig_val[idx]
        if numStain < p or numStain > 0:
            eig_vec = eig_vec[:, range(numStain)]
        score = np.dot(eig_vec.T, cov_mat)
        recon = np.dot(eig_vec, score) + np.average(image_2d)  # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
        #recon_img_mat = np.uint8(np.absolute(recon))  # TO CONTROL COMPLEX EIGENVALUES -----> to recontract rusulted image after reduce color

        return recon

    def startPCA(self, numStain=100):
        self.RGB_2_OD()
        r = self.od[:, :, 0]
        g = self.od[:, :, 1]
        b = self.od[:, :, 2]
        comp_r = self.getComponents(r, numStain)
        comp_g = self.getComponents(g, numStain)
        comp_b = self.getComponents(b, numStain)
        return np.dstack((comp_r, comp_g, comp_b))


if __name__=="__main__":
    img = plt.imread("../figure9.jpg")

    pca = PCA(img)
    resul = pca.startPCA(50)
    od = pca.getOD()

    plt.subplot(2, 4, 2)
    plt.imshow(img)

    plt.subplot(2, 4, 3)
    plt.imshow(od)

    plt.subplot(2, 3, 4)
    plt.imshow(resul[:,:,0], cmap="gray")

    plt.subplot(2, 3, 5)
    plt.imshow(resul[:, :, 1], cmap="gray")

    plt.subplot(2, 3, 6)
    plt.imshow(resul[:, :, 2], cmap="gray")
    plt.show()