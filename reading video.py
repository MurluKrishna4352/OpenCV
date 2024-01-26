import cv2
import numpy as np
import matplotlib as plt

framewidth = 640
frameheight = 360 

cap = cv2.VideoCapture("resources/test.mp4")
while True:
    sucess,img = cap.read()
    img = cv2.resize(img, (framewidth,frameheight))#resizing the video to desired values
    cv2.imshow("video",img)

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break #this will just come out of the loop when q key is pressed
    