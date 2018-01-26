import numpy as np
from matplotlib import pyplot as plt

def rgb_2_od(img):
    s = img.shape
    od = np.zeros(s)
    if (len(img)==3):
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[2]):
                    if img[i, j, k] != 0:
                        od[i, j, k] = np.log10(img[i, j, k])
    else:
        for i in range(s[0]):
            for j in range(s[1]):
                od[i, j] = np.log10(img[i, j])
    plt.title("Espace OD")
    plt.imshow(od)
    plt.show()
    return od
def rgb_2_od2(img):
    s = img.shape
    od = np.zeros(s)
    if (len(img)==3):
        for i in range(s[0]):
            for j in range(s[1]):
                for k in range(s[2]):
                    if img[i, j, k] != 0:
                        od[i, j, k] = np.log10(img[i, j, k])
    else:
        for i in range(s[0]):
            for j in range(s[1]):
                od[i, j] = np.log10(img[i, j])
    return od

if __name__ == "__main__":
    # reading the image
    # path="Tumor_CD31_LoRes.png"
    path = "../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"
    img = plt.imread(path)
    img = img[:, :, 0:3].astype(np.double)

    rgb_2_od(img)