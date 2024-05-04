import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0
 
# folder = "C:\sample_proj\python_projects\SIGN_LANGUAGE_DETECTION\Data\Hello"
folder = "C:\sample_proj\python_projects\SIGN_LANGUAGE_DETECTION\Data\Thank you"
# folder = "C:\sample_proj\python_projects\SIGN_LANGUAGE_DETECTION\Data\Yes"
# folder = "C:\sample_proj\python_projects\SIGN_LANGUAGE_DETECTION\Data\I Love You"
# folder = "C:\sample_proj\python_projects\SIGN_LANGUAGE_DETECTION\Data\I am happy"

while True:
    success , img = cap.read()
    hands , img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand ['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255  #to keep white background

        imgCrop = img[y-offset : y + h + offset , x-offset : x + w + offset]
        imgCropShape = imgCrop.shape

        aspectratio = h/w 

        if aspectratio > 1:  #dealing with width
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape =imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[: ,wGap : wCal +wGap] = imgResize

        else:
            k = imgSize / w     #dealing with height
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape =imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap : hCal +hGap, : ] = imgResize

        cv2.imshow('ImageCrop' , imgCrop)
        cv2.imshow('ImageWhite' , imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) #key of keyborad which we press to collect data
    if key == ord('s') : #select any key 's' to capture the data
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)   #format of image like jpd,png// folder will be created if not exists
        print(counter)
        