import numpy as np
import matplotlib.pyplot as plt
import OpticalDensity as od

class HSD:
    def __init__(self, img):
        self.img_0 = img
        self.img_hsi = None

    def RGB_2_GRAY(self):
        [l, c, d] = self.img_0.shape
        gray = np.zeros([l, c])
        for i in range(l):
            for j in range(c):
                gray[i, j] = sum(self.img_0[i, j])/3
        return gray

    def calcule_HSI(self):
        [l, c, d] = self.img_0.shape
        gray = self.RGB_2_GRAY()
        self.img_hsi = np.zeros([l, c, 3])
        self.img_hsi[:, :, 2] = gray
        for i in range(l):
            for j in range(c):
                x = (self.img_0[i, j, 0]/self.gray[i, j]) - 1
                y = (self.img_0[i, j, 1]-self.img_0[i, j, 2]) / (self.gray[i, j] * np.sqrt(3))
                self.img_hsi[i, j, 0] = self.getHue2(x, y)
                self.img_hsi[i, j, 1] = self.getSaturation2(x, y)

    def getSaturation2(self, cx, cy):
        return np.sqrt(np.square(cx)+np.square(cy))

    def getHue2(self, cx, cy):
        return np.arctan(cy/cx)

    def plotHSD(self):
        coord = self.getCoordinate()
        hue = []
        saturation = []
        for i in coord:
            hue.append(self.getHue2(i[0], i[1]))
            saturation.append(self.getSaturation2(i[0], i[1]))

        """
        from sklearn.neighbors import KNeighborsClassifier

        colors = ['#4EACC5', '#FF9C34', '#4E9A06']
        from sklearn.cluster import KMeans
        from sklearn.metrics.pairwise import pairwise_distances_argmin
        k_means = KMeans(n_clusters=3)
        k_means.fit(points)
        k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
        k_means_labels = pairwise_distances_argmin(points, k_means_cluster_centers)

        for k, col in zip(range(3), colors):
            my_members = k_means_labels == k
            cluster_center = k_means_cluster_centers[k]
            plt.plot(points[my_members, 0], points[my_members, 1], 'w', markerfacecolor=col, marker='.')
            #plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
        plt.title('KMeans')

        plt.show()"""

if __name__ == "__main__":
    # reading the image
    # path="Tumor_CD31_LoRes.png"
    path = "../DataSet/BreastCancerCell_dataset/ytma10_010704_benign1_ccd.tif"
    img = plt.imread(path)
    img = img[:, :, 0:3].astype(np.double)

    # hsi
    h = HSD(img)
    h.RGB_2_GRAY()
    h.calcule_HSI()
    h.plotHSD()

    #calculer la OD
    OD = od.rgb_2_od(img)

    # hsd
    h1 = HSD(OD)
    h1.RGB_2_GRAY()
    h1.calcule_HSI()
    h1.plotHSD()


