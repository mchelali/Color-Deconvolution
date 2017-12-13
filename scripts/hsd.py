#coding:  utf-8
import numpy as np
import matplotlib.pyplot as plt
import OpticalDensity as od
from sklearn.cluster import KMeans

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
        self.imgReconstruite=None
        self.intensity=None

    def getRed(self):
        return self.img_0[:,:,0]

    def getGreen(self):
        return self.img_0[:,:,1]

    def getBlue(self):
        return self.img_0[:,:,2]

    def GlobalIntensity(self):
        self.intensity = np.zeros([self.img_0.shape[0], self.img_0.shape[1]])
        for i in range(self.img_0.shape[0]):
            for j in range(self.img_0.shape[1]):
                self.intensity[i, j] = (self.getRed()[i,j]+ self.getGreen()[i,j] + self.getBlue()[i,j]) / 3

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

    def getCX(self,hue,saturation):
        return saturation*np.cos(hue)

    def getCY(self,hue,saturation):
        return saturation*np.sin(hue)

    def plotHSD(self):
        plt.subplot(1,3,1)
        plt.imshow(self.img_hsi[:,:,0], cmap="gray")

        plt.subplot(1,3,2)
        plt.imshow(self.img_hsi[:, :, 1], cmap="gray")

        plt.subplot(1, 3, 3)
        plt.imshow(self.img_hsi[:, :, 2], cmap="gray")
        plt.show()

    def plotHS(self,Z):
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if Z[i][j]==0:
                    c1 = plt.scatter(self.chroma[i,j,0], self.chroma[i,j, 1], c='r', marker='+')
                elif Z[i][j]==1:
                    c2 = plt.scatter(self.chroma[i, j, 0], self.chroma[i, j, 1], c='b', marker='o')
                elif Z[i][j]==2:
                    c3 = plt.scatter(self.chroma[i, j, 0], self.chroma[i, j, 1], c='g', marker='*')
                elif Z[i][j]==3:
                    c4 = plt.scatter(self.chroma[i, j, 0], self.chroma[i, j, 1], c='black', marker='l')

        plt.legend([c1, c2, c3], ['Cluster 0', 'Cluster 1','Cluster 2'])
        plt.title('K-means clusters into 3 clusters')
        plt.show()

    def plotHS2(self,Z):
        Z=np.reshape(Z,(self.chroma.shape[0]*self.chroma.shape[1],1))
        print Z.shape
        c1=0
        vec = np.reshape(self.chroma, (self.chroma.shape[0] * self.chroma.shape[1], 2))
        vec1 = []
        vec2 = []
        vec3 = []
        for i in range(Z.shape[0]):
            if Z[i]==0:
                vec1.append(vec[i])
            elif Z[i]==1:
                vec2.append(vec[i])
            elif Z[i]==2:
                vec3.append(vec[i])
        print len(vec1)

        c1 = plt.scatter(vec1[1:,0], vec1[1:,1], c='r', marker='+')

        c2 = plt.scatter(vec2[1:,0], vec2[1:,1], c='g', marker='o')

        c3 = plt.scatter(vec3[1:,0], vec3[1:,1], c='b', marker='*')
        plt.legend([c1, c2, c3], ['Cluster 0', 'Cluster 1', 'Cluster 2'])
        plt.title('K-means clusters into 3 clusters')
        plt.show()
        print "finish"


    def Kmeans(self,cluster):
        np.random.seed(42)
        vec=np.reshape(self.chroma, (self.chroma.shape[0]*self.chroma.shape[1],2))
        kmeans = KMeans(n_clusters=cluster, random_state=0)
        kmeans.fit(vec)
        kmeans.labels_=np.reshape(kmeans.labels_,(self.chroma.shape[0],self.chroma.shape[1]))
        Z=np.reshape(kmeans.labels_,self.chroma.shape[0]*self.chroma.shape[1],2)
        Z=np.reshape(Z[:100],(10,10))
        return Z

    def Kmeans2(self,cluster):
        np.random.seed(42)
        vec=np.reshape(self.chroma, (self.chroma.shape[0]*self.chroma.shape[1],2))
        kmeans = KMeans(n_clusters=cluster, random_state=0)
        kmeans.fit(vec)
        plt.scatter(vec[:1000, 0], vec[:1000, 1], c=kmeans.labels_[:1000])
        plt.show()

    def recontructionToRGB(self):
        #Calcul de intensity global de chaque pixel on en a besoin pour la reconstruction
        self.GlobalIntensity()
        [l,c,d]=self.img_hsi.shape
        self.imgReconstruite=np.zeros([l,c,3])
        for i in range(l):
            for j in range(c):
                cx = self.getCX(self.img_hsi[i, j, 0],self.img_hsi[i, j, 1])
                cy = self.getCY(self.img_hsi[i, j, 0],self.img_hsi[i, j, 1])

                self.imgReconstruite[i, j, 0] = self.intensity[i, j] * (cx + 1)
                self.imgReconstruite[i, j, 1] = 0.5 * self.intensity[i, j] * (2 - cx - np.sqrt(3) * cy)
                self.imgReconstruite[i, j, 2] = 0.5 * self.intensity[i, j] * (2 - cx - np.sqrt(3) * cy)
        plt.subplot(1,3,1)
        plt.title("hue")
        plt.imshow(self.imgReconstruite[:, :, 0], cmap="magma")
        plt.subplot(1,3,2)
        plt.title("saturation")
        plt.imshow(self.imgReconstruite[:, :, 1], cmap="RdPu")
        plt.subplot(1,3,3)
        plt.title("intensity or density")
        plt.imshow(self.imgReconstruite[:, :, 2], cmap="gray")

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
    h.Kmeans2(3)
    #h.plotHS(Z)
    h.recontructionToRGB()

    #calculer la OD
    OD = od.rgb_2_od(img)

    # hsd
    #h1 = HSD(OD)
    #h1.chromaticite()
    #h1.calcule_HSI()
    #h1.plotHSD()


