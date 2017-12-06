import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import fastica, PCA
import cv2

class PCA_:
    def __init__(self):
        """
        :param img: image d'une lame couleur RGB
        """
        # construction d'une matrice ou chaque colone represente un canal
        self.img = img
        self.cov_mat = None
        self.eig_val = None
        self.eig_vec = None

    def transformData(self, img):
        l, c, d = img.shape
        data = []
        for i in range(d):
            data.append(img[:, :, i].ravel())
        return np.array(data).T

    def getMean(self, data):
        print data.shape
        l, c = data.shape
        mean = []
        for i in range(l):
            mean.append(sum(data[i, :]) / c)
        return np.array(mean)

    def setImage(self, img):
        data = self.transformData(img)
        print("data shape ", data.shape)
        print("first line ", data[0, :])
        mean = self.getMean(data)
        print("mean shape ", mean.shape)
        print("first line ", mean[0])
        diff = np.zeros(data.shape)
        for i in range(data.shape[1]):
            diff[:, i] = data[:, i] - mean
        print("diff fisrt ", diff[0])
        num_data, dim = data.shape
        if dim > 100:
            print 'PCA - compact trick used'
            M = np.dot(diff, diff.T)  # covariance matrix
            e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors
            tmp = np.dot(diff.T, EV).T  # this is the compact trick
            V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
            S = np.sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
        else:
            print 'PCA - SVD used'
            U, S, V = np.linalg.svd(diff)
            V = V[:num_data]  # only makes sense to return the first num_data
        print V.shape
        # return the projection matrix, the variance and the mean
        return V, S, diff

    def getComponents(self, image_2d, numStain=100):
        data = self.transformData(image_2d)
        mean = self.getMean(data)
        diff = np.zeros(data.shape)
        for i in range(data.shape[1]):
            diff[:, i] = data[:, i] - mean
        print diff.shape
        self.cov_mat = np.cov(diff)
        # self.cov_mat = np.cov(diff)
        self.eig_val, self.eig_vec = np.linalg.eigh(
            self.cov_mat)  # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED
        p = len(self.eig_vec)
        idx = np.argsort(self.eig_val)  # sort array by idex
        idx = idx[::-1]  # reverse the sorted array
        self.eig_vec = self.eig_vec[:, idx]
        self.eig_val = self.eig_val[idx]
        if numStain < p or numStain > 0:
            self.eig_vec = self.eig_vec[:, range(numStain)]
        score = np.dot(self.eig_vec.T, diff)
        # print("score ",score.shape)
        # print("eig vec", self.eig_vec.shape)
        # recon = np.dot(self.eig_vec, score) + np.average(image_2d)  # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER
        """print 'PCA - compact trick used'
        M = dot(X,X.T) #covariance matrix
        e,EV = linalg.eigh(M) #eigenvalues and eigenvectors
        tmp = dot(X.T,EV).T #this is the compact trick
        V = tmp[::-1] #reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order"""
        # return np.absolute(recon)
        print self.eig_vec

    def getComponents2(self, img, numStain=100):
        data = self.transformData(img)
        pca = PCA()
        pca.fit(data)

def RGB_2_OD(img):
    [l, c, d] = img.shape
    od = np.zeros([l, c, d])
    for i in range(l):
        for j in range(c):
            for k in range(d):
                if img[i, j, k] != 0:
                    od[i, j, k] = np.log(img[i, j, k])
    return od


def lanch_pca(X):
    """"
    # Principal Component Analysis
    # input: X, matrix with training data as flattened arrays in rows
    # return: projection matrix (with important dimensions first),
    # variance and mean
    #
    """
    # get dimensions
    num_data, dim = X.shape

    # center data
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X

    if dim>100:
        print 'PCA - compact trick used'
        M = np.dot(X, X.T)  # covariance matrix
        e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors
        tmp = np.dot(X.T, EV).T  # this is the compact trick
        V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
        S = np.sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
    else:
        print 'PCA - SVD used'
        U, S, V = np.linalg.svd(X)
        V = V[:num_data] #only makes sense to return the first num_data
    print V.shape
    # return the projection matrix, the variance and the mean
    return V, S, mean_X


if __name__ == "__main__":
    img = plt.imread("../figure9.jpg")
    l,c, d = img.shape
    #img = cv2.resize(img, (c/3, l/3))
    print img.shape
    # print("taille image", img.shape)
    # print img.reshape(-1, 3).shape

    od = RGB_2_OD(img.astype(np.float))
    pca = PCA_()
    v, s, m = pca.setImage(od)
    print v
    print s
    print m

    #plt.imshow(od)
    #plt.show()
    #pca = PCA(n_components=3)
    #X_pca = pca.fit_transform(img.reshape(-1, 3))
    #print X_pca.shape
