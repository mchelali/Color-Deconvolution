
import matplotlib.pyplot as plt
from scripts import ColorDeconvolution


if __name__=="__main__":

    # reading the image
    #path="Tumor_CD31_LoRes.png"
    path="figure9.jpg"
    img = plt.imread(path)
    img = img[:, :, 0:3]

    satin = ColorDeconvolution(img)
    satin.separateStain()
    satin.showStains()
