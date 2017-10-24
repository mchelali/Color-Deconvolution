
import matplotlib.pyplot as plt
import numpy as np


img = plt.imread("figure9.jpg")

print img.shape
print len(img[:,:,0].ravel())

r = [range(10), range(10)]
g = [range(10), range(10)]
b = [range(10), range(10)]

print np.array([r,g,b]).transpose().shape
