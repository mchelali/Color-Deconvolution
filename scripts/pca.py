import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import fastica, PCA

class PCA_:
    def __init__(self, img):
        # constructeur de la class
        self.img_0 = img.astype(np.float)
        self.cov_mat = None
        self.eig_val = None
        self.eig_vec = None


    def getComponents(self, image_2d, numStain=100):
        diff = image_2d - np.average(image_2d)
        self.cov_mat = np.cov(diff)
        self.eig_val, self.eig_vec = np.linalg.eigh(self.cov_mat)  # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
        p = len(self.eig_vec)
        idx = np.argsort(self.eig_val) #sort array by idex
        idx = idx[::-1] #reverse the sorted array
        self.eig_vec = self.eig_vec[:, idx]
        self.eig_val = self.eig_val[idx]
        if numStain < p or numStain > 0:
            self.eig_vec = self.eig_vec[:, range(numStain)]
        score = np.dot(self.eig_vec.T, diff)
        #print("score ",score.shape)
        #print("eig vec", self.eig_vec.shape)
        recon = np.dot(self.eig_vec, score) + np.average(image_2d)  # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
        return np.absolute(recon)

    def startPCA(self, numStain=100):
        r = self.img_0
        comp_r = self.getComponents(r, numStain)
        return comp_r


def RGB_2_OD(img):
    [l, c, d] = img.shape
    od = np.zeros([l, c, d])
    for i in range(l):
        for j in range(c):
            for k in range(d):
                if img[i, j, k] != 0:
                    od[i, j, k] = np.log(img[i, j, k])
    return od

if __name__=="__main__":
    img = plt.imread("../figure9.jpg")
    #print("taille image", img.shape)

    od = RGB_2_OD(img.astype(np.float))
    pca_R = PCA_(od[:, :, 0])
    pca_G = PCA_(od[:, :, 1])
    pca_B = PCA_(od[:, :, 2])
    rest_R = pca_R.startPCA(3)
    rest_G = pca_G.startPCA(3)
    rest_B = pca_B.startPCA(3)

    #print("taille resultat", rest_R.shape)
    #print("matrice de covariance", pca_R.cov_mat)
    #print("matrice retourne apres pca", rest_R)

    #data = PCA()
    #data1 = data.fit_transform(img[:, :, 2])
    #data2 = data.inverse_transform(data1)
    #print data1

    #plt.subplot(2, 4, 1)
    #plt.imshow(data2, cmap="gray")

    plt.subplot(2, 4, 2)
    plt.imshow(img)

    plt.subplot(2, 4, 3)
    plt.imshow(od)

    plt.subplot(2, 3, 4)
    plt.imshow(rest_R, cmap="gray")

    plt.subplot(2, 3, 5)
    plt.imshow(rest_G, cmap="gray")

    plt.subplot(2, 3, 6)
    plt.imshow(rest_B, cmap="gray")

    plt.show()