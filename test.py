import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1,flipType=0)
detector.handDetector.maxHands = 1
detector.handDetector.minAreaSize = 500
classifier = None

offset = 20
imgSize = 300
counter = 0
labels = ["Hello","Thank you","I Love You"]


# Check if the model file and labels file exist
try:
    if not os.path.exists(r"C:\Users\Lenovo\Desktop\Model\keras_model.h5"):
        print("Model file not found. Please place 'keras_model.h5' in the same directory.")
    elif not os.path.exists(r"C:\Users\Lenovo\Desktop\Model\labels.txt"):
        print("Labels file not found. Please place 'labels.txt' in the same directory.")
    else:
        classifier = Classifier(r"C:\Users\Lenovo\Desktop\Model\keras_model.h5", r"C:\Users\Lenovo\Desktop\Model\labels.txt")
except Exception as e:
    print("An error occurred while checking the model and labels files:", str(e))





while True:
    success , img = cap.read()
    imgOutput = img.copy()
    hands , img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

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
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)

        else:
            k = imgSize / w     #dealing with height
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape =imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap : hCal +hGap, : ] = imgResize
            prediction, index = classifier.getPrediction(imgWhite,draw=False)
            
        cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x-offset+400,y-offset+60-50),(0,255,0),cv2.FILLED)    

        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
        cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h + offset),(0,255,0),4)
        
        cv2.imshow('ImageCrop' , imgCrop)
        cv2.imshow('ImageWhite' , imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
    
        