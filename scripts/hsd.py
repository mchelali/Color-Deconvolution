import numpy as np

class Point:
    def __init__(self, x=0, y=0):
        self.__cx = x
        self.__cy = y

    def setX(self, x):
        self.__cx=x

    def setY(self, y):
        self.__cy=y

    def getX(self):
        return self.__cx

    def getY(self):
        return self.__cy

    def getVector(self):
        return np.array([self.__cx, self.__cy])

class HSD:
    def __init__(self, img =None):
        self.img_0 = img
        self.od = None
        self.img_gray = None
        self.od_global = None
        self.img_hsi = None
        self.img_hsd = None

    def RGB_2_OD(self):
        [l, c, d] = self.img_0.shape
        self.od = np.zeros([l, c, d])
        for i in range(l):
            for j in range(c):
                for k in range(d):
                    if self.img_0[i, j, k] != 0:
                        self.od[i, j, k] = np.log(self.img_0[i, j, k])

    def RGB_2_GRAY(self):
        [l, c] = self.img_0.shape
        self.gray = np.zeros([l, c])
        for i in range(l):
            for j in range(c):
                self.gray[i, j] = sum(self.img_0[i, j])

    def calcule_HSI(self):
        if self.img_gray == None:
            print("have to calculate global intensity first")
        else:
            [l, c] = self.img_0.shape
            self.img_hsi = np.zeros([l,c])
            for i in range(l):
                for j in range(c):
                    x = (self.img_0[i, j, 0]/self.gray[i, j]) - 1
                    y = (self.img_0[i, j, 1]-self.img_0[i, j, 2]) / self.gray[i, j] * np.sqrt(3)
                    self.img_hsi[i, j] = Point(x, y)

    def calcule_HSD(self):
        if self.od_global == None:
            print("Have to calculate global OD first")
        else:
            [l, c] = self.img_gray.shape
            self.img_hsd = np.zeros([l,c])
            for i in range(l):
                for j in range(c):
                    x = (self.od[i, j, 0]/self.od_global[i, j]) - 1
                    y = (self.od[i, j, 1]-self.od[i, j, 2]) / self.od_global[i, j] * np.sqrt(3)
                    self.img_hsd[i, j] = Point(x, y)

    def getSaturation(self, point):
        n = 0
        for i in point.getVector():
            n = n + np.square(i)
        return np.sqrt(n)

    def getHue(self, point):
        return np.arctan(point.getY()/point.getX())
