# coding: utf8
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
                    self.od[i, j] = np.log10(self.img[i, j])

    def lanchePCA(self):
        # Application de la PCA
        pca = PCA(n_components=3)
        self.pca_ = pca.fit_transform(self.od)
        self.I_r =  np.dot(self.od.T, self.pca_)


    def lanchICA(self):
        # Application de ICA
        ica = FastICA(n_components=3, algorithm='parallel', whiten=True, fun='logcosh', max_iter=100,
                      tol=0.0001, w_init=None,random_state=2)
        self.ica_ = ica.fit_transform(self.pca_)
        self.P = np.dot(self.I_r, self.ica_.T)
        self.W = np.dot(self.pca_.T, self.P.T)


    def correctVectorICA(self, stain_number=3):
        # Initialisation du vecteur V_min ???
        v_min = np.zeros(self.ica_.shape)
        for i in range(self.ica_.shape[2]):
            v_min = self.ica_[:, i] / np.linalg.norm(self.ica_[:, i])
        print "v_min   ---->", v_min

        #init de la distance  d_i_j
        d = []
        for i in range(stain_number):
            d.append(0.1)
        d_min = 10**(-6)

        for i in range(self.ica_.shape[2]):
            # On calcule le cout: Cost(Vmin, P)
            cost = self.getCost(v_min[:, i])
            while(d[i] < d_min):
                #ici jai pas fait le calcule de U, cause de manque d'information
                V = v_min[:, i] + d[i]
                k = self.getCost(V)
                if k < cost :
                    v_min[:, i] = V
                    cost = k
                else:
                    d[i] = max(d[i]/10, d_min)
        return v_min

    def getCost(self, Vmin):
        v_min = np.zeros(self.ica_.shape)
        list_d = []
        for i in range(3):
            list_d.append( np.linalg.norm(self.P[:, i]) + ( np.dot(Vmin, self.P[:,i].T)**2 / np.linalg.norm(Vmin)**2) )



    def normalisation2(self, mat):
        mat = 255.0 * (mat - mat.min()) / (mat.max() - mat.min())
        return mat

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
    #img_path ="../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"
    #img_path = "../DataSet/arn1.tif"
    #img_path = "../Stain/Lung_4stain.tif"
    #img_path = "../Stain/Lung_4stain02.tif"
    #img_path = "../he.png"

    img_path="/home/ro0t34/Documents/projet/images/Nouveau dossier/imageOrig1.bmp"

    img = plt.imread(img_path)
    img = img[:, :, 0:3]
    l,c, d = img.shape
    #img = cv2.resize(img, (c/3, l/3))
    print img.shape
    stain = ICA_StainSeparation(img.astype(np.float))
    stain.RGB_2_OD()
    stain.lanchePCA()
    stain.lanchICA()
    stain.correctVectorICA(3)
    # Resahepe et visalisation des resultats
    img_pca = stain.normalisation2(stain.pca_).reshape((l, c, 3))
    img_ica = stain.normalisation2(stain.ica_).reshape((l, c, 3))

    fig = plt.figure(1, figsize=(15, 12))
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    plt.subplot(5, 3, 1)
    plt.title("PCA 1")
    plt.imshow(img_pca[:, :, 0], cmap="gray")

    plt.subplot(5, 3, 2)
    plt.title("PCA 2")
    plt.imshow(img_pca[:, :, 1], cmap="gray")

    plt.subplot(5, 3, 3)
    plt.title("PCA 3")
    plt.imshow(img_pca[:, :, 2], cmap="gray")

    pca1 = np.uint8(img_pca[:, :, 0])
    pca2 = np.uint8(img_pca[:, :, 1])
    pca3 = np.uint8(img_pca[:, :, 2])

    thres, pca1 = cv2.threshold(pca1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres2, pca2 = cv2.threshold(pca2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres3, pca3 = cv2.threshold(pca3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.subplot(5, 3, 4)
    plt.title("PCA 1")
    plt.imshow(pca1, cmap="gray")

    plt.subplot(5, 3, 5)
    plt.title("PCA 2")
    plt.imshow(pca2, cmap="gray")

    plt.subplot(5, 3, 6)
    plt.title("PCA 3")
    plt.imshow(pca3, cmap="gray")

    plt.subplot(5, 3, 7)
    plt.title("ICA 1")
    plt.imshow(img_ica[:, :, 0], cmap="gray")

    plt.subplot(5, 3, 8)
    plt.title("ICA 2")
    plt.imshow(img_ica[:, :, 1], cmap="gray")

    plt.subplot(5, 3, 9)
    plt.title("ICA 3")
    plt.imshow(img_ica[:, :, 2], cmap="gray")

    ica1 = np.uint8(img_ica[:, :, 0])
    ica2 = np.uint8(img_ica[:, :, 1])
    ica3 = np.uint8(img_ica[:, :, 2])

    thres, ica1 = cv2.threshold(ica1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres2, ica2 = cv2.threshold(ica2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thres3, ica3 = cv2.threshold(ica3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    plt.subplot(5, 3, 12)
    plt.title("ICA 1")
    plt.imshow(ica1, cmap="gray")

    plt.subplot(5, 3, 10)
    plt.title("ICA 2")
    plt.imshow(ica2, cmap="gray")

    plt.subplot(5, 3, 11)
    plt.title("ICA 3")
    plt.imshow(255 - ica3, cmap="gray")

    ica = cv2.merge((ica1, ica3, ica2))
    pca = cv2.merge((pca3, pca2, pca1))

    plt.subplot(5, 3, 13)
    plt.title("PCA ")
    plt.imshow(pca)

    plt.subplot(5, 3, 14)
    plt.title("ICA ")
    plt.imshow(ica)

    plt.subplot(5, 3, 15)
    plt.title("Original")
    plt.imshow(img)

    plt.show()

    # ce code permet de sauvegarder les images des méthodes, il faut juste donner le chemin
    import cv2
    path="/home/ro0t34/Pictures/"

    cv2.imwrite(path + "img.png", img)

    cv2.imwrite(path + "pca1.png", img_pca[:, :, 0])
    cv2.imwrite(path + "pca2.png", img_pca[:, :, 1])
    cv2.imwrite(path + "pca3.png", img_pca[:, :, 2])

    cv2.imwrite(path + "ica1.png", img_ica[:, :, 0])
    cv2.imwrite(path + "ica2.png", img_ica[:, :, 1])
    cv2.imwrite(path + "ica3.png", img_ica[:, :, 2])

    import Image
    img1 = Image.fromarray(img_ica[:, :, 0])
    print img1
    img1.save(path + "image.tiff")