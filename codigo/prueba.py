import numpy as np
import cv2 as cv
import datetime
import imutils
import argparse
import copy
import time
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from pyproj import Proj
from numpy.fft    import fft
from numpy.linalg import norm
from Inertial import Inertial
from Coordenadas import Coord
from Image import Image
from Pose import Pose
from skimage.measure import compare_ssim

#Crear Botones https://github.com/StanislavJochman/Kinematic_with_interpolation/blob/daba194ae1b7790559ebd37a15095216166c324b/functions.py#L24

def ZoomIn():
    pass

def GUI():
    #cv.createButton("Zoom In", ZoomIn, 0, cv.QT_PUSH_BUTTON,False)
    #cv.createTrackbar('Zoom','map',0,1,ZoomIn)
    pass


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
    width = widthBeam #metro cada pixel
    #scale_percent = 90# percent of original size
    #width = int(image.shape[1] * scale_percent / 100)
    #height = int(image.shape[0] * 100 / 100)
    height = 5 #la altura no varia
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
    #print(radian)
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

def rotate_line_pixel2(coordCont, radian): 
    x = float(coordCont[0])
    y = float(coordCont[1])
    radian = float(radian)

    xpos = int(x + np.cos(radian) * -94)
    ypos = int(y - np.sin(radian) * -94)
    #xpos = int(x * np.cos(radian) - y * np.sin(radian))
    #ypos = int(x * np.sin(radian) + y * np.cos(radian))
    
    return xpos,ypos


def rotate_line_pixel3(coordCont, radian): 
    x = coordCont[0]
    y = coordCont[1]
    radian = float(radian)
    xpos = int(x + np.cos(radian) * 94)
    ypos = int(y - np.sin(radian) * 94)
    
    return xpos,ypos


def rotate_line_pixel4(coordCont, radian, distance):
    print(counterLines, radian)
    x = coordCont[0]
    y = coordCont[1]
    radian = float(radian)
    xposIzq = int(np.cos(radian) * (distance) - np.sin(radian) * (0) + x);
    yposIzq = int(np.sin(radian) * (distance) + np.cos(radian) * (0) + y);
    xposDer = int(np.cos(radian) * (-distance) - np.sin(radian) * (0) + x);
    yposDer = int(np.sin(radian) * (-distance) + np.cos(radian) * (0) + y);   
    return xposIzq,yposIzq,xposDer,yposDer

def vector_projection(u, v):
    return (int) (np.dot(v, u)/np.dot(v, v))*v
    
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

def mapeoX(cmin, cmax, pmin, pmax):
    if(cmax == 0 and cmin == 0):
        return 0,0
    ax = (pmax-pmin)/(cmax-cmin)
    bx = (cmax*pmin-pmax*cmin)/(cmax-cmin)
    return ax,bx

def mapeoY(cmin, cmax, pmin, pmax):
    if(cmax == 0 and cmin == 0):
        return 0,0
    ay = (pmax-pmin)/(cmax-cmin)
    by = (cmax*pmin-pmax*cmin)/(cmax-cmin)
    return ay, by

def mapeaXY(ax, bx, ay, by, x, y):
    xp = ax*x+bx
    yp = ay*y+by
    return int(xp), int(yp)

def minCoord(xy):
    if(minXYm[0] > xy[0]):
        minXYm[0] = xy[0]
        print("cambio")
    if(minXYm[1] > xy[1]):
        minXYm[1] = xy[1]   
        print("cambio")
    
def maxCoord(xy):
    if(maxXYm[0] < xy[0]):
        maxXYm[0] = xy[0]
        print("cambio")
    if(maxXYm[1] < xy[1]):
        maxXYm[1] = xy[1]   
        print("cambio")

def humberto(xm, ym):
    kx = mapa.shape[1]/metros
    ky = mapa.shape[0]/metros
    offsetx =  mapa.shape[1]/2
    offsety =  mapa.shape[0]/2
    xp = np.abs(xm*kx-offsetx)
    xy = np.abs(ym*ky-offsety)
    return xp, xy

def rotation_symmetry(x, y, point): 
    x = x - point[0]
    y = y - point[1]
    y = -y
    x = np.abs(x + point[0])
    y = y + point[0]
    x, y = y, x
    return x,y

def remove_water_column(img):
    ret,thresh1 = cv.threshold(img,160,255,cv.THRESH_BINARY)
    #thresh1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,115,2)
    
    blanco = True
    for i in range(int(img.shape[1]/2),0,-1):
        if blanco and thresh1[0,i] == 255:
            img[0,i] = 0
            
        elif thresh1[0,i]==0:
            img[0,i] = 0
            blanco = False
        else:
            break
        
    blanco = True
    for i in range(int(img.shape[1]/2),int(img.shape[1])):
        if blanco and thresh1[0,i] == 255:
            img[0,i] = 0
            
        elif thresh1[0,i]==0:
            img[0,i] = 0
            blanco = False
        else:
            break
    return img

minXYm = [-600, -600]
maxXYm = [600, 600]

ax, bx = mapeoX(-600, 600, 0, 1023)
ay, by = mapeoY(-600, 600, 0, 1023)


##Datos sensores
widthBeam = 200
timeFrameNew = 3.3 #sec
apx = widthBeam*np.sin(60*np.pi/180)
apy = 200/widthBeam
metros = 200
horizontalBeamwidth = 0.5
verticalBeamwidth = 60
bufferInertial = Inertial.ReadFromFile("../recursos/datos/sibiu-pro-carboneras-anforas-2.jdb.salida")
bufferCoord    = Coord.ReadFromFile("../recursos/datos/coordenadas.txt")
bufferPose    = Pose.ReadFromFile("../recursos/datos/pose.txt")
cap = cv.VideoCapture('../recursos/datos/S200225_7.mp4')
coordInit  = bufferCoord[0]
mapa = np.zeros((1024, 1024,1), dtype = "uint8")
mapa.fill(0) # or img[:] = 255
mapa2 = np.zeros((1024, 1024,1), dtype = "uint8")
mapa2.fill(0) # or img[:] = 255
sift = cv.xfeatures2d.SIFT_create()
matcher = cv.BFMatcher()  # buscador de coincidencias por fuerza bruta
contador = 0
bufferImages = []
inicioMapa = [mapa.shape[1]-1,0]
finMapa    = [mapa.shape[1]-1,mapa.shape[0]-1]
centroMapa = [mapa.shape[0]/2, mapa.shape[1]/2]
lineaPrev = None

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")


cv.namedWindow("map")


GUI()
playVideo = True
primera = True
listFrames = [] #Contiene cada frame
listFramesNews = [] #Contiene todos los pixeles nuevos
indexFrameBack  = 0
indexFrame = 0
indexFrameROI = 0
indiceROIX = 511
indiceROIY = 0
counterLines=0
last_crop_img = None
cropImg = None
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame

      #ret, frame = cap.read()
      #listFramesNews.append(frame)
      #milli_time = float((time.time()))
      #primera = False
  if(playVideo):
      
      ret, frame = cap.read()
      listFrames.append(frame)
      row, col, _ = frame.shape
      if(primera is True):
          milli_time = float(time.time())
          listFramesNews.append(frame)
          primera = False
      current_milli_time = float(time.time())
      resultTime = current_milli_time-milli_time
      if(resultTime >= 3.3):
          milli_time = float((time.time()))
          listFramesNews.append(frame)
          print(('nuevo frame'))
          
  if ret == True:
    # Display the resulting frame
    cv.imshow('Frame',frame)
    
    k = cv.waitKeyEx(25)
    """
    if len(listFrames) > 1:
        imageA = listFrames[len(listFrames)-2]
        imageB = frame
        grayA = cv.cvtColor(imageA, cv.COLOR_BGR2GRAY)
        grayB = cv.cvtColor(imageB, cv.COLOR_BGR2GRAY)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
        diff = (diff * 255).astype("uint8")
        print("SSIM: {}".format(score))
        thresh = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        cnts = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        for c in cnts:
            	# compute the bounding box of the contour and then draw the
            	# bounding box on both input images to represent where the two
            	# images differ
            	(x, y, w, h) = cv.boundingRect(c)
            	cv.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
            	cv.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # show the output images
        cv.imshow("Original", imageA)
        cv.imshow("Modified", imageB)
        cv.imshow("Diff", diff)
        cv.imshow("Thresh", thresh)
        """     
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
        img = listFramesNews[indexFrameROI]
        lineaPixels = img[indiceROIX:indiceROIX+1, 0:col]
        lineaPixels = adjust_light(lineaPixels)
        cropImg = copy.copy(lineaPixels)
        
        #lineaPixels = resize_image(lineaPixels)
        #lineaPixels = cv.cvtColor(lineaPixels, cv.COLOR_BGR2GRAY)
        
        xm, ym = float(bufferCoord[counterLines].x), float(bufferCoord[counterLines].y)
        xMetrosCentral, yMetrosCentral = int((xm-float(coordInit.x))*50), int((ym-float(coordInit.y))*50)        
        
        xMetrosIzquierda, yMetrosIzquierda  = xMetrosCentral-100, yMetrosCentral
        xMetrosDerecha, yMetrosDerecha      = xMetrosCentral+100, yMetrosCentral
        
        ax, bx = mapeoX(minXYm[0], maxXYm[1], 0, 1023)
        ay, by = mapeoY(minXYm[1], maxXYm[1], 0, 1023)
        xPixelCentral,yPixelCentral     = mapeaXY(ax, bx, ay, by, xMetrosCentral, yMetrosCentral)
        xPixelIzquierda,yPixelIzquierda = mapeaXY(ax, bx, ay, by, 100, 0)
        


 

        #https://www.youtube.com/watch?v=C8LoyjinqD0 mapping
        #Rotamos los puntos 90 grados respecto al centro (512,512)
        #https://www.youtube.com/watch?v=nu2MR1RoFsA
        
        
        xPixelCentral, yPixelCentral     = rotation_symmetry(xPixelCentral,   yPixelCentral,    (mapa.shape[1]/2, mapa.shape[0]/2))

        xPixelDistancia = np.abs(xPixelIzquierda-513) #Solo necesito uno para calcular que 100 metros son X pixels.
        xIzq,yIzq,xDer,yDer = rotate_line_pixel4((xPixelCentral, yPixelCentral),float(bufferPose[counterLines].yaw), xPixelDistancia)
        mapa[int(xPixelCentral),int(yPixelCentral)] = 255
        mapa[xIzq, yIzq] = 255
        mapa[xDer, yDer] = 255
        
        puntoPintar = bresenham_algorithm((xIzq, yIzq), (xDer, yDer))
        widthBeam = len(puntoPintar)
        bufferImages.append(Image(counterLines, lineaPixels, bufferCoord[counterLines].y, (xIzq, yIzq, xDer, yDer), widthBeam)) #Almacenamos la imagen para computar posibles intersecciones
        colorGrayLineaPixels = cv.cvtColor(lineaPixels, cv.COLOR_BGR2GRAY)
        remove_water_column(colorGrayLineaPixels)
        resizeLineaPixels = resize_image(colorGrayLineaPixels)
        cv.imshow('map2', resizeLineaPixels)
        
        mapaPintarBoolean = []
        """
        for i, pair in enumerate(puntoPintar):
            for j in range(0,5):
                if mapa[pair[0]+j,pair[1]] != 0 :
                    mapaPintarBoolean.append((pair[0],pair[1],j, True))
                else:
                    mapaPintarBoolean.append((pair[0],pair[1],j, False))

        i = 0
        for _, pair in enumerate(mapaPintarBoolean):
            if(pair[3]):
                mapa[pair[0]+pair[2],pair[1]] = (resizeLineaPixels[pair[2], i]+mapa[pair[0]+pair[2],pair[1]])/2
            else:
                mapa[pair[0]+pair[2],pair[1]] = resizeLineaPixels[pair[2], i]
            if pair[2] == 4: # Hardcore
                i = i +1
            #while(1):
            #   k = cv.waitKey(33)
            #   if k==27:    # Esc key to stop
            #       break     
        """

        counterLines = counterLines + 1   
        indiceROIX = indiceROIX - 1
        if(indiceROIX==-1):
            indexFrameROI = indexFrameROI+1
            indiceROIX = row-1 #Solo procesamos la primera linea
            print("cambio frame")
            
            
        cv.imshow('map', mapa)
        
        if(last_crop_img is None): 
            last_crop_img = copy.copy(cropImg)
        if(counterLines > 1):
            last_crop_img = np.concatenate((lineaPrev, last_crop_img), axis=0)
            cv.imshow('concatenate',last_crop_img)
        lineaPrev = copy.copy(cropImg)
        #cv.imshow('map2', mapa2)
        

  # Break the loop
  else: 
    break

#importante https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
