from __future__ import division
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2


class ColorSpace:
    def __init__(self,img):
        self.img_0=img
        self.l=self.img_0.shape[0]
        self.c=self.img_0.shape[1]
        self.img_norm=np.zeros((self.l,self.c,3))
        self.hsv=np.zeros((self.l,self.c,3))
        self.hsl=np.zeros((self.l,self.c,3))
        self.hsv_conique=np.zeros((self.l,self.c,3))
        self.hsl_conique=np.zeros((self.l,self.c,3))

    def normalise(self):
        r = self.img_0[:, :, 0]
        g = self.img_0[:, :, 1]
        b = self.img_0[:, :, 2]

        max_r=np.amax(r)
        max_g=np.amax(g)
        max_b=np.amax(b)




        self.img_norm[:, :, 0] = self.img_0[:, :, 0] / max_r
        self.img_norm[:, :, 1] = self.img_0[:, :, 1] / max_g
        self.img_norm[:, :, 2] = self.img_0[:, :, 2] / max_b


        return self.img_norm
    #s,v entre 0 et 1 h entre -1 et 5 (je pense qu'il faut normaliser entre 0 et klk chose)
    def HSV(self):
        r=self.img_norm[:,:,0]
        g=self.img_norm[:,:,1]
        b=self.img_norm[:,:,2]

        r=r.ravel()
        g=g.ravel()
        b=b.ravel()

        hue=np.zeros(len(r))
        saturation = np.zeros(len(g))
        value = np.zeros(len(b))
        for i in range(len(r)):
            #lightness = v dans leurs methodes
            value[i]=max(r[i],g[i],b[i])
            saturation[i]=0 if value[i]==0 else (value[i]-min(r[i],g[i],b[i]))/value[i]

            if max(r[i],g[i],b[i])==r[i]:
                hue[i]=0 if (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))==0 else ((g[i] - b[i]) / (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i])))

            elif max(r[i],g[i],b[i])== g[i]:
                hue[i] = (b[i] - r[i]) / (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))
            elif max(r[i],g[i],b[i])== b[i]:
                hue[i] = (r[i] - g[i]) / (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))

        hue = np.reshape(hue,(self.l,self.c))
        saturation = np.reshape(saturation,(self.l,self.c))
        value = np.reshape(value,(self.l,self.c))
        self.hsv[:,:,0]= hue
        self.hsv[:,:,1]= saturation
        self.hsv[:,:,2]= value

        return self.hsv

    def HSL(self):
        r = self.img_norm[:, :, 0]
        g = self.img_norm[:, :, 1]
        b = self.img_norm[:, :, 2]

        r = r.ravel()
        g = g.ravel()
        b = b.ravel()

        hue = np.zeros(len(r))
        saturation = np.zeros(len(g))
        lightness = np.zeros(len(b))
        for i in range(len(hue)):
            lightness[i]=(max(r[i],g[i],b[i])+min(r[i],g[i],b[i]))/2
            if lightness[i] <= 0.5:
                saturation[i]=0 if (max(r[i], g[i], b[i]) + min(r[i], g[i], b[i]))==0 else (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i])) / (
                    max(r[i], g[i], b[i]) + min(r[i], g[i], b[i]))
            else:
                saturation[i]=0 if (2 - max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))==0 else (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i])) / (2 - max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))
            if (max(r[i],g[i],b[i])==r[i]):
                hue[i]=0 if (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))==0 else (g[i] - b[i]) / (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))


            elif (max(r[i],g[i],b[i])== g[i]):
                hue[i] = (b[i] - r[i]) / (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))

            elif (max(r[i],g[i],b[i])== b[i]):
                hue[i] = (r[i] - g[i]) / (max(r[i], g[i], b[i]) - min(r[i], g[i], b[i]))

        hue = np.reshape(hue,(self.l,self.c))
        saturation = np.reshape(saturation,(self.l,self.c))
        lightness = np.reshape(lightness,(self.l,self.c))

        self.hsl[:, :, 0] = hue
        self.hsl[:, :, 1] = saturation
        self.hsl[:, :, 2] = lightness

        return self.hsl

    def transformationConiqueHSV(self):

        self.hsv_conique=self.hsv
        saturation=self.hsv[:,:,1].ravel()
        value=self.hsv[:,:,2].ravel()
        for i in range(len(saturation)):
            saturation[i]=saturation[i]*value[i]
        saturation=np.reshape(saturation,(self.l,self.c))
        self.hsv_conique[:,:,1]=saturation
        return self.hsv_conique

    def transformationConiqueHSL(self):

        self.hsl_conique = self.hsl
        saturation = self.hsl[:, :, 1].ravel()
        lightness = self.hsl[:, :, 2].ravel()
        for i in range(len(saturation)):
            saturation[i] = saturation[i]*(1-2*abs(0.5-lightness[i]))
        saturation = np.reshape(saturation, (self.l, self.c))
        self.hsl_conique[:, :, 1] = saturation
        return self.hsl_conique


    #codage des formules jusqua la partie 9 il reste la 10 et 11 dans la prochaine fonction
    #formule de passage en norme l1
    def luminanceSaturationTeinteL1(self):
        img=np.zeros((self.l,self.c,3))
        img=np.reshape(img,(self.l*self.c,3))
        self.img_norm=np.reshape(self.img_norm,(self.l*self.c,3))

        for i in range(len(self.img_norm)):
            r = self.img_norm[i, 0]
            g = self.img_norm[i, 1]
            b = self.img_norm[i, 2]
            trie = []
            trie.append(r)
            trie.append(g)
            trie.append(b)
            trie.sort()
            max = trie[2]
            mid = trie[1]
            min = trie[0]
            if (r == b == g):

                m1 = 1 / 3 * (r + g + b)
                s1 = 1 / 2 * (r + g - 2 * b)
                phi = (-(b + r - 2 * g) / 2 * s1) + 1 / 2
                #print "pareille"
            else:
                # equation 21
                # luminance
                m1 = 1 / 3 * (r + g + b)

                # saturation
                # equation 16 et 21
                if (b + r >= 2 * g or m1 >= g):
                    s1 = 3 / 2 * (r - m1)
                # equation 17 et 21
                elif (b + r <= 2 * g or m1 <= g):
                    s1 = 3 / 2 * (m1 - b)

                # equation 13
                sigma1 = 1 / 3 * (abs(2 * r - g - b) + abs(2 * g - b - r) + abs(2 * b - r - g))
                # equation 18
                k = s1 / sigma1

                # Teinte 18 et 19
                if (r + b >= 2 * g):
                    x = float( (k / s1) * (g - b))
                    if math.isnan(x):
                        teinte = 0
                    else:
                        teinte = (k / s1) * (g - b)

                elif (r + b <= 2 * g):
                    teinte = 1 - (3 / 4 * ((r - g) / s1))

                # equation 21
                phi= 0 if (r * s1) ==0 else 1 / 2 - ((b + r - 2 * g) / r * s1)


            # equation 22 et 23
            if (b + r >= 2 * g):
                r1 = m1 + (2 / 3) * s1
                g1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * phi
                b1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * phi
            elif (r + b <= 2 * g):
                r1 = m1 + s1 - (2 / 3) * s1 * phi
                g1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 * phi
                b1 = m1 - 2 / 3 * s1
            x = float(r1)
            r1=0 if (math.isinf(float(r1))) else r1

            img[i, 0] = r1
            img[i, 1] = g1
            img[i, 2] = b1

        img = np.reshape(img,(self.l, self.c, 3))
        self.img_norm = np.reshape(self.img_norm,(self.l, self.c, 3))

        img1=np.zeros((self.l,self.c,3))
        img1[:, :, 0] = img[:, :, 0]
        img1[:, :, 1] = img[:, :, 1]
        img1[:, :, 2] = img[:, :, 2]

        img=self.normalise2(img)
        print "-----------SUCCESS norme L1---------------"
        plt.subplot(1, 4, 1)
        plt.title("")
        plt.imshow((img[:, :, 0].astype(np.uint8)), cmap="gray")
        plt.subplot(1, 4, 2)
        plt.imshow((img[:, :, 1].astype(np.uint8)), cmap="gray")
        plt.subplot(1, 4, 3)
        plt.imshow((img[:, :, 2].astype(np.uint8)), cmap="gray")
        plt.subplot(1, 4, 4)
        plt.imshow(img1)
        plt.show()
        self.binarisation(img)
        return img

    def passageAuCubeDigitalComplet(self):
        img = np.zeros((self.l, self.c, 3))
        img = np.reshape(img,(self.l*self.c,3))
        self.img_norm = np.reshape(self.img_norm, (self.l * self.c, 3))
        for i in range(len(self.img_norm)):
            r = self.img_norm[i, 0]
            g = self.img_norm[i, 1]
            b = self.img_norm[i, 2]
            trie = []
            trie.append(r)
            trie.append(g)
            trie.append(b)
            trie.sort()

            max = trie[2]
            mid = trie[1]
            min = trie[0]
            # remplacement du systeme 21 par celui ci
            # system 25
            # passageAuCubeDigitalComplet

            if (r > g >= b):
                lambdaa = 0

            elif (g >= r > b):
                lambdaa = 1

            elif (g > b >= r):
                lambdaa = 2
            elif (b >= g > r):
                lambdaa = 3
            elif (b > r >= g):
                lambdaa = 4

            elif (r >= b > g):

                lambdaa = 5

            # ceci remplace l'equation 21
            m1 = 1 / 3 * (max + mid + min)

            if (max + min >= 2 * mid):
                s1 = 3 / 2 * (max - m1)
            elif (max + min <= 2 * mid):
                s1 = 3 / 2 * (m1 - min)

            h1 = 42*(lambdaa + 1/2 -((-1)**lambdaa)*((max+min-2*mid)/2*s1))

            phi = (h1/42) - lambdaa

            # equation 22 et 23
            # je pense que cest la reconstruction
            if lambdaa == 0:
                # a verfier si cest 42*phi ou bien la valeur de h1 qui est inf a 24
                if (42*phi <= 24):
                    r1 = m1 + (2 / 3) * s1
                    g1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                    b1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                elif (42*phi >= 24):
                    r1 = m1 + s1 - (2 / 3) * s1 * 42*phi
                    g1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 *42*phi
                    b1 = m1 - 2 / 3 * s1

            elif lambdaa == 1:
                if (42*phi <= 24):
                    g1 = m1 + (2 / 3) * s1
                    r1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                    b1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                elif (42*phi >= 24):
                    g1 = m1 + s1 - (2 / 3) * s1 * 42*phi
                    r1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 *42*phi
                    b1 = m1 - 2 / 3 * s1

            elif lambdaa == 2:
                if (42*phi <= 24):
                    g1 = m1 + (2 / 3) * s1
                    b1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                    r1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                elif (42*phi >= 24):
                    g1 = m1 + s1 - (2 / 3) * s1 * 42*phi
                    b1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 * 42*phi
                    r1 = m1 - 2 / 3 * s1


            elif lambdaa == 3:
                if (42*phi <= 24):
                    b1 = m1 + (2 / 3) * s1
                    g1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                    r1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                elif (42*phi >= 24):
                    b1 = m1 + s1 - (2 / 3) * s1 * 42*phi
                    g1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 *42*phi
                    r1 = m1 - 2 / 3 * s1

            elif lambdaa == 4:
                if (42*phi <= 24):
                    b1 = m1 + (2 / 3) * s1
                    r1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                    g1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                elif (42*phi >= 24):
                    b1 = m1 + s1 - (2 / 3) * s1 * 42*phi
                    r1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 *42*phi
                    g1 = m1 - 2 / 3 * s1

            elif lambdaa == 5:
                if (42*phi <= 24):
                    r1 = m1 + (2 / 3) * s1
                    b1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                    g1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * 42*phi
                elif (42*phi >= 24):
                    r1 = m1 + s1 - (2 / 3) * s1 * 42*phi
                    b1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 *42*phi
                    g1 = m1 - 2 / 3 * s1
            img[i, 0] = r1
            img[i, 1] = g1
            img[i, 2] = b1
        img = np.reshape(img, (self.l, self.c, 3))
        self.img_norm = np.reshape(self.img_norm, (self.l, self.c, 3))
        img1 = np.zeros((self.l, self.c, 3))
        img1[:, :, 0] = img[:, :, 0]
        img1[:, :, 1] = img[:, :, 1]
        img1[:, :, 2] = img[:, :, 2]
        print "-----------SUCCESS--Cube-DigitalComplet-------------"
        img = self.normalise2(img)
        plt.subplot(1,3,1)
        plt.imshow((img[:,:,0].astype(np.uint8)),cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow((img[:, :, 1].astype(np.uint8)), cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow((img[:, :, 2].astype(np.uint8)), cmap="gray")


        #plt.title("PassageAuCubeDigitalComplet")
        plt.show()
        self.binarisation(img)


        return img

    # passer de la representation cylindrique a conique
    # remplacer la saturation hsl par max-min
    def EssainormMaxMin(self):
        self.hsl_conique=np.reshape(self.hsl_conique,(self.l*self.c,3))
        img=np.zeros((self.l,self.c,3))
        img=np.reshape(img,(self.l*self.c,3))
        for i in range(len(self.hsl_conique)):
            r = self.hsl_conique[i, 0]
            g = self.hsl_conique[i, 1]
            b = self.hsl_conique[i, 2]
            trie = []
            trie.append(r)
            trie.append(g)
            trie.append(b)
            trie.sort()

            max = trie[2]
            mid = trie[1]
            min = trie[0]
            if (r == b == g):
                #print True
                m1 = 1 / 3 * (r + g + b)
                #s1 = max - min
                s1 = 1 / 2 * (r + g - 2 * b)
                phi = (-(b + r - 2 * g) / 2 * s1) + 1 / 2
            else:
                # equation 21
                # luminance
                m1 = 1 / 3 * (r + g + b)

                # saturation
                s1 = max-min

                # equation 13
                sigma1 = 1 / 3 * (abs(2 * r - g - b) + abs(2 * g - b - r) + abs(2 * b - r - g))
                # equation 18
                k = s1 / sigma1

                # Teinte 18 et 19
                if (r + b >= 2 * g):
                    teinte = k / s1 * (g - b)
                elif (r + b <= 2 * g):
                    teinte = 1 - (3 / 4 * ((r - g) / s1))

                # equation 21
                phi = 1/2 - ((b + r - 2 * g) / r * s1)

            # equation 22 et 23
            # je pense que cest la reconstruction
            if (b + r >= 2 * g):
                r1 = m1 + (2 / 3) * s1
                g1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * phi
                b1 = m1 - (1 / 3) * s1 + (2 / 3) * s1 * phi
            elif (r + b <= 2 * g):
                r1 = m1 + s1 - (2 / 3) * s1 * phi
                g1 = m1 - (1 / 3) * s1 + 2 / 3 * s1 * phi
                b1 = m1 - 2 / 3 * s1
            img[i, 0]= r1
            img[i, 1]= g1
            img[i, 2]= b1

        img = np.reshape(img,(self.l,self.c,3))
        self.hsl_conique = np.reshape(self.hsl_conique, (self.l, self.c, 3))
        print "-----------SUCCESS---------------"
        #img=self.normalise2(img)
        plt.subplot(1, 3, 1)
        plt.imshow(img[:, :, 0]*255, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.imshow(img[:, :, 1]*255, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.imshow(img[:, :, 2]*255, cmap="gray")
        plt.title("Essai norm max min")
        plt.show()
        return img

    # cette methode est la bonne a mon avis
    # passer de la representation cylindrique a conique
    # remplacer la saturation hsl par max-min
    def normMaxMin(self):
        hsl_conique = np.zeros((self.l,self.c,3))
        saturation = self.hsl[:, :, 1].ravel()
        self.hsl=np.reshape(self.hsl,(self.l*self.c,3))


        for i in range(self.l*self.c):
            r = self.hsl[i, 0]
            g = self.hsl[i, 1]
            b = self.hsl[i, 2]

            trie = []
            trie.append(r)
            trie.append(g)
            trie.append(b)
            trie.sort()

            max = trie[2]
            mid = trie[1]
            min = trie[0]

            saturation[i] = max-min


            x = float(saturation[i])
            if math.isnan(x):
                saturation[i]=0
                print True, i

        self.hsl = np.reshape(self.hsl, (self.l , self.c, 3))
        saturation = np.reshape(saturation, (self.l, self.c))

        hsl_conique[:, :, 1] = saturation
        hsl_conique[:,:,0] = self.hsl[:,:,0]
        hsl_conique[:, :, 2] = self.hsl[:, :, 2]
        img=self.normalise2(hsl_conique)


        print "-------------norm-Max-Min-----Done--------"

        plt.subplot(1, 3, 1)
        plt.title("Hue")
        plt.imshow((img[:, :, 0].astype(np.uint8)), cmap="gray")

        plt.subplot(1, 3, 2)
        plt.title("Saturation")
        plt.imshow((img[:, :, 1].astype(np.uint8)), cmap="gray")

        plt.subplot(1, 3, 3)
        plt.title("Luminance")
        plt.imshow((img[:, :, 2].astype(np.uint8)), cmap="gray")
        plt.show()
        self.binarisation(img)

        return hsl_conique

    def normalise2(self,img):
        r=img[:, :, 0]
        g=img[:, :, 1]
        b=img[:, :, 2]

        r = r.ravel()
        b = b.ravel()
        g = g.ravel()

        I0=np.amin(r)
        I1=np.amax(r)

        I00 = np.amin(g)
        I11 = np.amax(g)

        I000 = np.amin(b)
        I111 = np.amax(b)

        imgg=np.zeros((self.l,self.c,3))
        print "--------------normalise process [0,255] --------------------"
        for i in range(len(r)):

            r[i] = ((255*r[i]) / (I1 - I0)) - ((255*I0) / (I1 - I0))
            g[i] = ((255*g[i]) / (I11 - I00)) - ((255*I00) / (I11 - I00))
            b[i] = ((255 * b[i]) / (I111 - I000)) - ((255 * I000) / (I111 - I000))


        r = np.reshape(r, (self.l, self.c))
        g = np.reshape(g, (self.l, self.c))
        b = np.reshape(b, (self.l, self.c))

        imgg[:, :, 0] = r
        imgg[:, :, 1] = g
        imgg[:, :, 2] = b
        print "--------------normalise succeed-----------------"

        return imgg
    def binarisation(self,imgReconstruite):

        # Otsu's thresholding after Gaussian filtering
        # il faut lui donner image self.imgReconstruite et binariser chaque canal hue saturation et density

        blur1 = cv2.GaussianBlur((imgReconstruite[:, :, 0]*255).astype(np.uint8), (5, 5), 0)
        ret1, th1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        blur2 = cv2.GaussianBlur((imgReconstruite[:, :, 1]*255).astype(np.uint8), (5, 5), 0)
        ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        blur3 = cv2.GaussianBlur((imgReconstruite[:, :, 2]*255).astype(np.uint8), (5, 5), 0)
        ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        plt.subplot(1,3,1)
        plt.title("hue")
        plt.imshow(th1,cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Saturation")
        plt.imshow(th2,cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title("")
        plt.imshow(th3,cmap="gray")

        plt.show()
        return th3


if __name__ == '__main__':
    x=input("tell me")
    if x==1:
        path1 = "C:/Users/ismet/Desktop/Final/Color-Deconvolution/Resultat/arn1.tif/HSD/HSI_Density.tif"
        path2 = "C:/Users/ismet/Desktop/Final/Color-Deconvolution/Resultat/arn1.tif/HSD/HSI_Saturation.tif"
        path3 = "C:/Users/ismet/Desktop/Final/Color-Deconvolution/Resultat/arn1.tif/HSD/HSI_Teinte.tif"

        img1 = plt.imread(path1)
        img2 = plt.imread(path2)
        img3 = plt.imread(path3)
        img = np.zeros([img1.shape[0], img1.shape[1], 3])
        img[:, :, 0] = img3[:, :]
        img[:, :, 1] = img2[:, :]
        img[:, :, 2] = img1[:, :]
    else:
        path1="C:/Users/ismet/Desktop/Final/Color-Deconvolution/DataSet/BreastCancerCell_dataset/ytma10_010704_malignant3_ccd.tif"
        #path1="C:/Users/ismet/Desktop/Final/Color-Deconvolution/Resultat/tumor.png/tumor.png"
        img = plt.imread(path1)

    color=ColorSpace(img)
    color.normalise()
    #imgg=color.normalise2(img)

    img_hsv = color.HSV()
    img_hsl = color.HSL()
    color.transformationConiqueHSV()
    color.transformationConiqueHSL()
    color.luminanceSaturationTeinteL1()
    color.passageAuCubeDigitalComplet()
    color.normMaxMin()
