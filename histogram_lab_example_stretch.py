import cv2
import numpy as np


x = np.random.rand(3,4,4)
x = cv2.imread('home1.jpg')
y = cv2.resize(x,None, fx=1.5, fy =1.5, interpolation = cv2.INTER_LINEAR)
z = y[0:150,0:150,:]
print(x.shape)
print(y.shape)
print(z.shape)