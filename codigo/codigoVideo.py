import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import datetime
import imutils
import argparse
from Inertial import Inertial

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv.warpAffine(image, M, (nW, nH))


##Datos sensores
bufferInertial = Inertial.ReadFromFile("../recursos/datos/sibiu-pro-carboneras-anforas-2.jdb.salida")
mapa = np.zeros((800, 600,1), dtype = "uint8")


cap = cv.VideoCapture('../recursos/datos/S200225_7.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

playVideo = True
primera = True
listFrames = []
indexFrameBack  = 0
indexFrame = 0
indexFrameROI = 0
indiceROIX = 0
indiceROIY = 0
counterLines=0
last_crop_img = None
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  if(playVideo):
      ret, frame = cap.read()
      listFrames.append(frame)
      if(primera):
          row, col, _ = frame.shape
          primera = False
  if ret == True:
    # Display the resulting frame
    cv.imshow('Frame',frame)
    k = cv.waitKeyEx(25)
    # Press Q on keyboard to  exit
    if k & 0xFF == ord('q'):
      break
    # Press Space on keyboard to pause/play
    elif k & 0xFF == ord('p'):
      if(not playVideo):
        playVideo = True
        indexFrameBack = 0
      else:
        playVideo = False
    # Press right arrow for go to frame to frame
    elif k == 2555904:
        if(playVideo) :
          playVideo = False
        else:
          if(indexFrameBack > 0):
            indexFrameBack -= 1
            frame = listFrames[len(listFrames)-1-indexFrameBack]
            cv.imshow('Frame',frame)
          else:
            _, frame = cap.read()
            listFrames.append(frame)
          
            cv.imshow('Frame',frame)
    # Press left arrow for go to frame to frame back
    elif k == 2424832:
        if(playVideo) :
          playVideo = False
        else:
          if(indexFrameBack+1 < len(listFrames)):
            indexFrameBack += 1
            frame = listFrames[len(listFrames)-1-indexFrameBack]
          cv.imshow('Frame',frame)
    elif k == ord('s'):
        fname = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        cv.imwrite(fname+'.png',frame)
        
        #https://likegeeks.com/es/procesar-de-imagenes-en-python/
    elif k & 0xFF == ord('n'):   
        img = listFrames[indexFrameROI]
        crop_img = img[indiceROIX:indiceROIX+1, 0:col]
        crop_img = imutils.rotate_bound(crop_img, bufferInertial[counterLines].euler[2])
        if(last_crop_img is not None ):
            #crop_img = cv.vconcat([last_crop_img, crop_img])
            crop_img = np.concatenate((last_crop_img, crop_img), axis=0)
        last_crop_img = crop_img
        indiceROIX +=1
        if(indiceROIX==row):
            indexFrameROI = indexFrameROI+1   
            indiceROIX = 0
            last_crop_img = None
        print(bufferInertial[counterLines].euler)
        counterLines = counterLines + 1
        cv.imshow('Frame2', crop_img)

  # Break the loop
  else: 
    break


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
