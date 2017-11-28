
import matplotlib.pyplot as plt
from scripts.ColorDeconvolution import ColorDeconvolution
from scripts.hsd import HSD


if __name__=="__main__":

    # reading the image
    #path="Tumor_CD31_LoRes.png"
    path="figure9.jpg"
    img = plt.imread(path)
    img = img[:, :, 0:3]


    # deconvolution de couleur
    satin = ColorDeconvolution(img)
    satin.RGB_2_OD()
    satin.separateStain()
    satin.showStains()
    """

    #hsd

    h = HSD(img)
    h.RGB_2_OD()
    h.OD_GLOBAL()
    h.calcule_HSD()
    h.plotHSD()
    """