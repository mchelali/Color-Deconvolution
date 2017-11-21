import numpy as np

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
    return od