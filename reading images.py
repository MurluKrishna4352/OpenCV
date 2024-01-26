import cv2
import numpy as np
import matplotlib as plt
#reading of image
img = cv2.imread('resources/lena.png')
cv2.imshow("Lena",img)
cv2.waitKey(0)

