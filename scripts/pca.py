import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
import cv2

class ICA_StainSeparation:
    def __init__(self, img):
        """
        :param img: image d'une lame couleur RGB
        """
        # construction d'une matrice ou chaque colone represente un canal
        self.size = img.shape
        self.img = img.reshape(-1, 3)
        self.od = None
        self.pca_ = None # Matrice reduite de PCA
        self.ica_ = None # Matrice reduite de ICA

    def RGB_2_OD(self):
        self.od = np.zeros([self.size[0]*self.size[1], self.size[2]])
        for i in range(self.size[0]*self.size[1]):
            for j in range(self.size[2]):
                if self.img[i, j] != 0:
                    self.od[i, j] = np.log(self.img[i, j])

    def lanchePCA(self):
        # Application de la PCA
        pca = PCA(n_components=3)
        self.pca_ = pca.fit_transform(self.od)
        self.I_r =  np.dot(self.od.T, self.pca_)
        print "I_r : ",self.I_r

    def lanchICA(self):
        # Application de ICA
        ica = FastICA(n_components=3, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=200,
                      tol=0.0001, w_init=None, random_state=None)
        self.ica_ = ica.fit_transform(self.pca_)
        self.P = np.dot(self.I_r, self.ica_.T)
        print "P : ", self.P
        self.W = np.dot(self.pca_.T, self.P.T)
        print "W : ", self.W

    def correctVectorICA(self, stain_number=2):
        # Initialisation du vecteur V_min ???

        #init de la distance  d_i_j
        d=[]
        for i in range(stain_number):
            d.append(0.1)

        #on calcule le cout
        cost = sum(d)/stain_number

    # ---------------------------------------------Test---------------------------------------------------#
    def transformData(self, img):                                                                 #-------#
        l, c, d = img.shape                                                                       #-------#
        data = []                                                                                 #-------#
        for i in range(d):                                                                        #-------#
            data.append(img[:, :, i].ravel())                                                     #-------#
        return np.array(data).T                                                                   #-------#
                                                                                                  #-------#
    def getMean(self, data):                                                                      #-------#
        print data.shape                                                                          #-------#
        l, c = data.shape                                                                         #-------#
        mean = []                                                                                 #-------#
        for i in range(l):                                                                        #-------#
            mean.append(sum(data[i, :]) / c)                                                      #-------#
        return np.array(mean)                                                                     #-------#
                                                                                                  #-------#
    def setImage(self, img):                                                                      #-------#
        data = self.transformData(img)                                                            #-------#
        print("data shape ", data.shape)                                                          #-------#
        print("first line ", data[0, :])                                                          #-------#
        mean = self.getMean(data)                                                                 #-------#
        print("mean shape ", mean.shape)                                                          #-------#
        print("first line ", mean[0])                                                             #-------#
        diff = np.zeros(data.shape)                                                               #-------#
        for i in range(data.shape[1]):                                                            #-------#
            diff[:, i] = data[:, i] - mean                                                        #-------#
        print("diff fisrt ", diff[0])                                                             #-------#
        num_data, dim = data.shape                                                                #-------#
        if dim > 100:                                                                             #-------#
            print 'PCA - compact trick used'                                                      #-------#
            M = np.dot(diff, diff.T)  # covariance matrix                                         #-------#
            e, EV = np.linalg.eigh(M)  # eigenvalues and eigenvectors                             #-------#
            tmp = np.dot(diff.T, EV).T  # this is the compact trick                               #-------#
            V = tmp[::-1]  # reverse since last eigenvectors are the ones we want                 #-------#
            S = np.sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order             #-------#
        else:                                                                                     #-------#
            print 'PCA - SVD used'                                                                #-------#
            U, S, V = np.linalg.svd(diff)                                                         #-------#
            V = V[:num_data]  # only makes sense to return the first num_data                     #-------#
        print V.shape                                                                             #-------#
        # return the projection matrix, the variance and the mean                                 #-------#
        return V, S, diff                                                                         #-------#
                                                                                                  #-------#
    def getComponents(self, image_2d, numStain=100):                                              #-------#
        data = self.transformData(image_2d)                                                       #-------#
        mean = self.getMean(data)                                                                 #-------#
        diff = np.zeros(data.shape)                                                               #-------#
        for i in range(data.shape[1]):                                                            #-------#
            diff[:, i] = data[:, i] - mean                                                        #-------#
        print diff.shape                                                                          #-------#
        self.cov_mat = np.cov(diff)                                                               #-------#
        # self.cov_mat = np.cov(diff)                                                             #-------#
        # USING "eigh", SO THAT PROPRTIES OF HERMITIAN MATRIX CAN BE USED                         #-------#
        self.eig_val, self.eig_vec = np.linalg.eigh( self.cov_mat)                                #-------#
        p = len(self.eig_vec)                                                                     #-------#
        idx = np.argsort(self.eig_val)  # sort array by idex                                      #-------#
        idx = idx[::-1]  # reverse the sorted array                                               #-------#
        self.eig_vec = self.eig_vec[:, idx]                                                       #-------#
        self.eig_val = self.eig_val[idx]                                                          #-------#
        if numStain < p or numStain > 0:                                                          #-------#                                                                      #-------#
            self.eig_vec = self.eig_vec[:, range(numStain)]                                       #-------#
        score = np.dot(self.eig_vec.T, diff)                                                      #-------#                                                                      #-------#                                                                      #-------#
        # print("score ",score.shape)                                                             #-------#
        # print("eig vec", self.eig_vec.shape)                                                    #-------#
        # recon = np.dot(self.eig_vec, score) + np.average(image_2d)  # SOME NORMALIZATION CAN BE USED TO MAKE IMAGE QUALITY BETTER                                                                      #-------#
        """print 'PCA - compact trick used'                                                        #-------#
        M = dot(X,X.T) #covariance matrix                                                          #-------#
        e,EV = linalg.eigh(M) #eigenvalues and eigenvectors                                        #-------#
        tmp = dot(X.T,EV).T #this is the compact trick                                             #-------#
        V = tmp[::-1] #reverse since last eigenvectors are the ones we want                        #-------#
        S = sqrt(e)[::-1] #reverse since eigenvalues are in increasing order"""                    #-------#
        # return np.absolute(recon)                                                                #-------#
        print self.eig_vec                                                                         #-------#
                                                                                                   #-------#
    def getComponents2(self, img, numStain=100):                                                   #-------#
        data = self.transformData(img)                                                             #-------#
        pca = PCA()                                                                                #-------#
        pca.fit(data)                                                                              #-------#
# ---------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    img = plt.imread("../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif")
    l,c, d = img.shape
    #img = cv2.resize(img, (c/3, l/3))
    print img.shape
    stain = ICA_StainSeparation(img.astype(np.float))
    stain.RGB_2_OD()
    stain.lanchePCA()
    stain.lanchICA()
    # Resahepe et visalisation des resultats
    img_pca = stain.pca_.reshape((l, c, 3))
    img_ica = stain.ica_.reshape((l, c, 3))

    plt.subplot(3, 1, 1)
    plt.title("Original")
    plt.imshow(img)

    plt.subplot(3, 3, 4)
    plt.title("ICA R")
    plt.imshow(img_ica[:, :, 0], cmap="gray")

    plt.subplot(3, 3, 5)
    plt.title("ICA G")
    plt.imshow(img_ica[:, :, 1], cmap="gray")

    plt.subplot(3, 3, 6)
    plt.title("ICA B")
    plt.imshow(img_ica[:, :, 2], cmap="gray")

    plt.subplot(3, 3, 7)
    plt.title("PCA R")
    plt.imshow(img_pca[:, :, 0], cmap="gray")

    plt.subplot(3, 3, 8)
    plt.title("PCA G")
    plt.imshow(img_pca[:, :, 1], cmap="gray")

    plt.subplot(3, 3, 9)
    plt.title("PCA B")
    plt.imshow(img_pca[:, :, 2], cmap="gray")

    plt.show()

