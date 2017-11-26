import matplotlib.pyplot as plt
import numpy as np

class PCA_:
    def __init__(self, img):
        """
        :param img: image d'une lame couleur RGB
        """
        # construction d'une matrice ou chaque colone represente un canal
        self.img_0 = img.reshape(-1, 3).T
        self.cov_mat = None
        self.eig_val = None
        self.eig_vec = None


    def getComponents(self, image_2d, numStain=100):
        diff = image_2d - image_2d.mean(axis=0)
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
        #recon = np.dot(self.eig_vec, score) + np.average(image_2d)  # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
        """print 'PCA - compact trick used'
        M = dot(X,X.T) #covariance matrix
        e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T #this is the compact trick
        V = tmp[::-1] #reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order"""
        #return np.absolute(recon)

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
    print img.reshape(-1, 3).shape
    #od = RGB_2_OD(img.astype(np.float))
    #pca = PCA_(od)

