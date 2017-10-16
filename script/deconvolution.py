#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np


def RGB_2_OD(img):
    [l, c, d] = img.shape
    od = np.zeros([l, c, d])
    for i in range(l):
        for j in range(c):
            for k in range(d):
                if img[i,j,k] != 0 :
                    od[i,j,k] = np.log(img[i, j, k])
    return od
def norm(vector):
    n = 0
    for i in vector:
        n = n + np.square(i)
    return np.sqrt(n)

def separateStain(img, mask):
    [l,c,d] = img.shape
    stains = np.zeros([l, c, d])
    for i in range(l):
        for j in range(c):
            a = np.dot(img[i, j], mask)
            stains[i, j, 0] = a[0]
            stains[i, j, 1] = a[1]
            stains[i, j, 2] = a[2]
    return stains

def normalization(img):
    [l,c,d] = img.shape
    #imageOut(:,:,i) = (imageOut(:,:,i)-min(Ch(:)))/(max(Ch(:)-min(Ch(:))));
    norm = np.zeros([l,c,d])
    for i in range(d):
        norm[:,:,i] = (img[:,:,i]-img.min()) / (img.max()-img.min())
    return norm

if __name__=="__main__":

    # set of standard values for stain vectors (from python scikit)
    #He = [0.65; 0.70; 0.29];
    #Eo = [0.07; 0.99; 0.11];
    #DAB = [0.27; 0.57; 0.78];

    # alternative set of standard values (HDAB from Fiji)
    He = np.array([ 0.6500286,  0.704031,    0.2860126 ]) # Hematoxylin
    Eo = np.array([0.07, 0.99, 0.11]) #Eosine
    DAB = np.array([ 0.26814753,  0.57031375,  0.77642715]) # DAB
    Res = np.array([ 0.7110272, 0.42318153, 0.5615672 ]) #residual


    # combine stain vectors to deconvolution matrix
    HDABtoRGB = np.array([He/norm(He), Eo/norm(Eo), DAB/norm(DAB)])
    RGBtoHDAB = np.linalg.inv(HDABtoRGB)

    print HDABtoRGB
    print "---"
    print RGBtoHDAB

    # reading the image
    path="Tumor_CD31_LoRes.png"
    #path="figure9.jpg"
    img = plt.imread(path)
    img = img[:, :, 0:3]
    img1 = RGB_2_OD(img)
    print img.shape
    stains = separateStain(img1.astype(np.double), RGBtoHDAB)
    #stains = normalization(stains)
    print stains.max()

    plt.subplot(1,4,1)
    plt.title("original")
    plt.imshow(img)

    plt.subplot(1,4,2)
    plt.title('Hematoxylin')
    plt.imshow(stains[:,:,0], cmap="gray")

    plt.subplot(1, 4, 3)
    plt.title('Eosine')
    plt.imshow(stains[:, :, 1], cmap="gray")

    plt.subplot(1, 4, 4)
    plt.title('DAB')
    plt.imshow(stains[:, :, 2], cmap="gray")

    plt.show()
