import cv2
import numpy as np
import matplotlib as plt

cap = cv2.VideoCapture(0)# zero is for the default camera
while True:
    success,img = cap.read()
    cv2.imshow("camera",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break