#coding:  utf-8
import numpy as np
import matplotlib.pyplot as plt
import OpticalDensity as od
from sklearn.cluster import KMeans
from matplotlib import cm
import cv2
from PIL import Image

class HSD:
    def __init__(self, img,path,type):
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
        self.path=path
        self.path2="C:\Users\ismet\Desktop\Final\Color-Deconvolution\Resultat"
        self.type=type
        self.imageBinariser=None

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

                self.chroma[i, j ,0] = (self.img_0[i, j, 0] / gray[i, j]) - 1 if (gray[i, j] - 1) != 0 else 0
                self.chroma[i, j, 1] = (self.img_0[i, j, 1] - self.img_0[i, j, 2]) / (gray[i, j] * np.sqrt(3)) if (gray[i, j] * np.sqrt(3)) !=0 else 0


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
        return 0 if cx==0 else np.arctan(cy/cx)

    def getCX(self,hue,saturation):
        return saturation*np.cos(hue)

    def getCY(self,hue,saturation):
        return saturation*np.sin(hue)
    # plotHSD permet de afficher les images de chaque canal de la hsd en gray
    def plotHSD(self):
        plt.subplot(1,3,1)
        plt.imshow(self.img_hsi[:,:,0], cmap="gray")

        plt.subplot(1,3,2)
        plt.imshow(self.img_hsi[:, :, 1], cmap="gray")

        plt.subplot(1, 3, 3)
        plt.imshow(self.img_hsi[:, :, 2], cmap="gray")
        plt.show()
    # Afficher le plot manuellement selon les valeurs de chroma
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

    def binarisation(self):

        # Otsu's thresholding after Gaussian filtering
        # il faut lui donner image self.imgReconstruite et binariser chaque canal hue saturation et density


        blur1 = cv2.GaussianBlur((self.imgReconstruite[:, :, 0]*255).astype(np.uint8), (5, 5), 0)
        ret1, th1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        blur2 = cv2.GaussianBlur((self.imgReconstruite[:, :, 1]*255).astype(np.uint8), (5, 5), 0)
        ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        blur3 = cv2.GaussianBlur((self.imgReconstruite[:, :, 2]*255).astype(np.uint8), (5, 5), 0)
        ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        plt.subplot(1,3,1)
        plt.title("hue")
        plt.imshow(th1,cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Saturation")
        plt.imshow(th2,cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title(self.type)
        plt.imshow(th3,cmap="gray")


        plt.show()
        self.saveOpencv(th1,"Hue_Binariser.png")
        self.saveOpencv(th2, "Saturation_Binariser.png")
        self.saveOpencv(th3, "Intensity_Binariser.png")
        return th3

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

        plt.subplot(2,3,1)
        plt.title("hue")
        plt.imshow(self.imgReconstruite[:, :, 0], cmap="gray")

        self.saveOpencv((self.imgReconstruite[:, :, 0]*255).astype(np.uint8), "HSI_Teinte.png")
        #plt.colorbar(ticks=[0, 60, 120, 179], orientation='horizontal', cmap=cm.hsv)

        plt.subplot(2,3,2)
        plt.title("saturation")
        plt.imshow(self.imgReconstruite[:, :, 1], cmap="gray")

        # save HSI
        self.saveOpencv((self.imgReconstruite[:, :, 1]*255).astype(np.uint8), "HSI_Saturation.png")

        plt.subplot(2,3,3)
        plt.title(self.type)
        plt.imshow(self.imgReconstruite[:, :, 2], cmap="gray")

        # save HSD
        #self.savePillow((self.imgReconstruite[:,:,2]*255).astype(np.uint8), self.path+"HSD_Density.tif")
        self.saveOpencv((self.imgReconstruite[:,:,2]*255).astype(np.uint8), "HSI_Density.png")


        #img = Image.fromarray(self.img_0[:, :, 0])
        #img.save(self.path + "img.tif")

        plt.show()

    def saveOpencv(self,img,path):
        cv2.imwrite(self.path2+"/"+self.path[19:]+"/"+path, img)

    def savePillow(self,img,path):
        img_to_save = Image.fromarray(img)
        img_to_save.save(self.path2+"/"+self.path[19:]+"/"+path)



if __name__ == "__main__":
    # reading the image
    # path="Tumor_CD31_LoRes.png"
    #path = "../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"

    path = "../DataSet_Lomenie/tumor.png"
    img = plt.imread(path)
    img = img[:, :, 0:3].astype(np.double)
    od=od.rgb_2_od2(img)

    # hsi
    h = HSD(od, path, "Density")
    h.chromaticite()
    h.calcule_HSI()
    # h.Kmeans2(3)
    # h.plotHS(Z)
    h.recontructionToRGB()
    h.binarisation()

