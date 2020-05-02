import numpy as np
import cv2 as cv
import datetime
import imutils
import argparse
import copy
from matplotlib import pyplot as plt

from numpy.fft    import fft
from numpy.linalg import norm
from Inertial import Inertial
from Coordenadas import Coord
from Image import Image

def write_matrix_to_textfile(a_matrix, file_to_write):

    def compile_row_string(a_row):
        return str(a_row).strip(']').strip('[').replace(' ','')

    with open(file_to_write, 'w') as f:
        for row in a_matrix:
            f.write(compile_row_string(row)+'\n')

    return True

def readrgb(image):
    return cv.cvtColor( image, cv.COLOR_BGR2RGB) 

def rgb2gray(x):
    return cv.cvtColor(x,cv.COLOR_RGB2GRAY)

def extractContours(g):
    #gt = (g > 128).astype(np.uint8)*255
    (_, gt) = cv.threshold(g,128,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    #gt = cv.adaptiveThreshold(g,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,101,-10)
    (_, contours, _) = cv.findContours(gt.copy(), cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    return [ c.reshape(-1,2) for c in contours ]

def invar(c, wmax=10):
    x,y = c.T
    z = x+y*1j
    f  = fft(z)
    fa = abs(f)                     # para conseguir invarianza a rotación 
                                    # y punto de partida
    s = fa[1] + fa[-1]              # el tamaño global de la figura para normalizar la escala
    v = np.zeros(2*wmax+1)          # espacio para el resultado
    v[:wmax] = fa[2:wmax+2];        # cogemos las componentes de baja frecuencia, positivas
    v[wmax:] = fa[-wmax-1:];        # y negativas. Añadimos también la frecuencia -1, que tiene
                                    # que ver con la "redondez" global de la figura
   
    if fa[-1] > fa[1]:              # normalizamos el sentido de recorrido
        v[:-1] = v[-2::-1]
        v[-1] = fa[1]
    
    return v / s


def razonable(c):
    return cv.arcLength(c, closed=True) >= 50

def scaling(image_width,in_scale, out_scale):
    # compute scaling values for input and output images
    in_width = int(image_width*in_scale)
    window_size = (in_width*3, in_width*3) # this should be a large canvas, used to create container size
    out_width = int(window_size[0]*out_scale)
    window_shift = [in_width/2, in_width/2]
    return (in_width, out_width, window_size, window_shift)

def adjust_light(image):
    clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(8,8))
    lab_image = cv.cvtColor(image, cv.COLOR_RGB2Lab)
    lab_planes = cv.split(lab_image)
    lab_planes[0] = clahe.apply(lab_planes[0])
    # Merge the the color planes back into an Lab image
    lab_image = cv.merge(lab_planes,lab_planes[0])
    #cv2.imshow("lab_image after clahe", lab_image);
    # convert back to RGB space and store the color corrected image
    result = cv.cvtColor(lab_image, cv.COLOR_Lab2RGB)
    return result

def resize_image(image):
    scale_percent = 60 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * 100 / 100)
    dim = (width, height)
    return cv.resize(image, dim, interpolation = cv.INTER_AREA)

def rotate_image(image, yaw):
    scaleFactor = 1.0 
    (oldH, oldW, oldC) = image.shape
    rotationMatrix2D = cv.getRotationMatrix2D(center=(oldW/2, oldH/2), angle = -yaw, scale = scaleFactor)
    [newX, newY] = oldW * scaleFactor, oldH * scaleFactor
    r = np.deg2rad(yaw)
    [newX, newY] = (abs(np.sin(r)*newY) + abs(np.cos(r)*newX),abs(np.sin(r)*newX) + abs(np.cos(r)*newY))
    # find the translation that moves the result to the center of that region.
    (tx, ty) = ((newX-oldW)/2, (newY-oldH)/2)
    rotationMatrix2D[0, 2] += tx # third column of matrix holds translation, which takes effect after rotation.
    rotationMatrix2D[1, 2] += ty
    rotated_img = cv.warpAffine(image,rotationMatrix2D, dsize=(int(newX), int(newY)))
    return rotated_img, rotationMatrix2D

def extract_features(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints_sift, descriptors = sift.detectAndCompute(gray, None)
    return (keypoints_sift, descriptors)


def matching_features(img1, img2):
    if len(bufferImages) > 1:
        img1 = bufferImages[len(bufferImages)-2]
        img2 = bufferImages[len(bufferImages)-1]
        des1 = img1.descrp
        des2 = img2.descrp
        kp1 = img1.kpts
        kp2 = img2.kpts
        matches = matcher.knnMatch(des1, des2, k = 2)
        print(matches)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow('Map',img3)
        
        
def find_contour(image):
    # Grayscale 
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
      
    # Find Canny edges 
    _, binary = cv.threshold(gray, 225, 255, cv.THRESH_BINARY_INV)

      
    # Finding Contours 
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image 
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 
      

      
    print("Number of Contours found = " + str(len(contours))) 
      
    # Draw all contours 
    # -1 signifies drawing all contours 
    cv.drawContours(image, contours, -1, (0, 255, 0), 3) 
      
    cv.imshow('Contours', image) 

    




#https://cristianpb.github.io/blog/image-rotation-opencv
def rotate_line_pixel(coordCont, degree):
    
    radian = (-degree*np.pi)/180
    print(radian)
    pair = coordCont[0][0]
    x1 = pair[0] * np.cos(radian) - pair[1] * np.sin(radian)
    y1 = pair[0] * np.sin(radian) + pair[1] * np.cos(radian)
    pair = coordCont[0][1]
    x2 = pair[0] * np.cos(radian) - pair[1] * np.sin(radian)
    y2 = pair[0] * np.sin(radian) + pair[1] * np.cos(radian)
   
    
    x1y1 =np.array([x1, y1])
    x2y2 =np.array([x2, y2])



    coord = [np.array([x1y1,x2y2], dtype=np.int32)]


    return coord

def vector_projection(u, v):
    return (int) (np.dot(u, v)/np.dot(v, v))*v
    
def scalar_projection(u, v):
    return (int) (np.dot(u, v) / np.linalg.norm(v))
        


#https://stackoverflow.com/questions/16028752/how-do-i-get-all-the-points-between-two-point-objects
def bresenham_algorithm(v1, v2):
    vResult, incYi, incXi, incYr, incXr =[], 0, 0, 0, 0
    dY = (v2[1] - v1[1])
    dX = (v2[0] - v1[0])
    #Incrementos para secciones con avance inclinado
    if(dY >= 0):
        incYi = 1
    else:
        dY = -dY
        incYi = -1
        
    if(dX >= 0):
        incXi = 1
    else:
        dX = -dX
        incXi = -1
    
    #Incrementos para secciones con avance recto   
    if(dX >= dY):
        incYr = 0
        incXr = incXi
    else:
        incXr = 0
        incYr = incYi
        #Cuando dy es mayor que dx, se intercambian, para reutilizar el mismo bucle. Intercambio rapido de variables en python
        #k = dX, dX = dY, dY = k
        dX, dY = dY, dX
        
    # Inicializar valores (y de error).
    x, y = v1[0], v1[1]
    avR = (2 * dY)
    av = (avR - dX)
    avI = (av - dX)
    vResult.append([x,y]) #podemos pintar directamente
    while x != v2[0]:
        if(av >= 0):
            x = (x + incXi)
            y = (y + incYi)
            av = (av + avI)
        else:
            x = (x + incXr)
            y = (y + incYr)
            av = (av + avR)
        vResult.append([x,y]) #podemos pintar directamente
    
    return vResult;




##Datos sensores
bufferInertial = Inertial.ReadFromFile("../recursos/datos/sibiu-pro-carboneras-anforas-2.jdb.salida")
bufferCoord    = Coord.ReadFromFile("../recursos/datos/coordenadas.txt")
cap = cv.VideoCapture('../recursos/datos/S200225_7.mp4')
coordInit  = bufferCoord[0]
mapa = np.zeros((800, 600,1), dtype = "uint8")
mapa.fill(0) # or img[:] = 255
sift = cv.xfeatures2d.SIFT_create()
matcher = cv.BFMatcher()  # buscador de coincidencias por fuerza bruta
contador = 0
bufferImages = []

inicioMapa = [799,0]
finMapa    = [799,599]

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

playVideo = True
primera = True
listFrames = []
indexFrameBack  = 0
indexFrame = 0
indexFrameROI = 0
indiceROIX = 511
indiceROIY = 0
counterLines=0
last_crop_img = None
last_crop_img_color = None
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
        lineaPixels = img[indiceROIX:indiceROIX+1, 0:col]
        cv.imshow('Frame2', lineaPixels)
        lineaPixels = adjust_light(lineaPixels)
        lineaPixels = resize_image(lineaPixels)
        lineaPixels = cv.cvtColor(lineaPixels, cv.COLOR_BGR2GRAY)
        radian = bufferInertial[counterLines].euler[2]*np.pi/180
        (y, x) = lineaPixels.shape
        box = [np.array([[0,0],[x-1,0]], dtype=np.int32)]
        box2 = [np.array([[0,inicioMapa[0]],[finMapa[1],finMapa[0]]], dtype=np.int32)]
        #print(box[0][1] - box[0][0])
        #print(box2[0][1] - box2[0][0])
        #https://math.stackexchange.com/questions/175896/finding-a-point-along-a-line-a-certain-distance-away-from-another-point
        #Para comprobar si tenemos frente a nosotros tres puntos colineales, debemos calcular la distancia entre cada punto extremo y el intermedio, y verificar si la suma de ambos valores es igual a la distancia entre los puntos extremos.
        #https://definicion.de/puntos-colineales/
        #https://stackoverflow.com/questions/38213717/how-to-find-the-points-along-two-vectors-where-the-distance-is-equal-to-x
        #print(box)
        #box = rotate_box(box, bufferInertial[counterLines].euler[2] )
        #print(box2)
        #print(y, x)
        vectorLineaXY = (box[0][1] - box[0][0])
        vectorMapaXY = (box2[0][1] - box2[0][0])
        x1y1Linea = box[0][0]
        x1y1Mapa = box2[0][0]
        proyeccionEscalar = scalar_projection(vectorLineaXY, vectorMapaXY)
        proyeccionVectorial = vector_projection(vectorLineaXY, vectorMapaXY)

        
        box2 = rotate_line_pixel(box2, bufferInertial[counterLines].euler[2] )
        puntoPintar = bresenham_algorithm(box2[0][0], box2[0][1])

        for i, pair in enumerate(puntoPintar):
            #print(pair[1]-1, pair[0]-1)
            if i >= x-1: break
            mapa[pair[1], pair[0]] = lineaPixels[0,i]
            #while(1):
            #   k = cv.waitKey(33)
            #   if k==27:    # Esc key to stop
            #       break            
        indiceROIX = indiceROIX - 1
        if(indiceROIX==row):
            indexFrameROI = indexFrameROI+1   
            indiceROIX = 544
            last_crop_img = None
        #print(bufferInertial[counterLines].euler)
        counterLines = counterLines + 1
        inicioMapa[0]  =  inicioMapa[0] - 1
        finMapa[0]     =  finMapa[0] - 1
        cv.imshow('map', mapa)
        cv.imshow('Frame2', lineaPixels)


  # Break the loop
  else: 
    break

#importante https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
