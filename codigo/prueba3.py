import numpy as np
import cv2 as cv
import datetime
import copy
import time
import threading
from ventana import *
from matplotlib import pyplot as plt
from numpy.fft    import fft
from Inertial import Inertial
from Velocity import Velocity
from Coordenadas import Coord
from Image import Image
from Pose import Pose

SIZE_BEAM_METERS = 15*2
SIZE_BEAM_PIXELS = 908
SIZE_SCALE_FACTOR = 10

bufferInertial  = Inertial.ReadFromFile("../recursos/datos/sibiu-pro-carboneras-anforas-2.jdb.salida")
#bufferCoord     = Coord.ReadFromFile("../recursos/datos/coordenadas.txt")
bufferCoord     = Coord.ReadFromFile("../recursos/datos/datosposicion.txt")
bufferPose      = Pose.ReadFromFile("../recursos/datos/pose.txt")
bufferVelocity  = Velocity.ReadFromFile("../recursos/datos/velocity.txt")
cap             = cv.VideoCapture('../recursos/datos/S200225_7.mp4')


mapa = np.zeros((4000, 4000, 3), dtype = "uint8")
mapa.fill(0) # or img[:] = 255
mapa = cv.cvtColor(mapa, cv.COLOR_RGB2BGR)
height = 1




def windowUI():
    global window, app
    app = QtWidgets.QApplication([])
    window = Ventana()

    window.show()
    app.exec_()

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

def resize_image(image, width, height):
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
    x = coordCont[0]
    y = coordCont[1]
    radian = float(radian)
    xposIzq = int(np.cos(radian) * (distance) - np.sin(radian) * (0) + x)
    yposIzq = int(np.sin(radian) * (distance) + np.cos(radian) * (0) + y)
    xposDer = int(np.cos(radian) * (-distance) - np.sin(radian) * (0) + x)
    yposDer = int(np.sin(radian) * (-distance) + np.cos(radian) * (0) + y)
    return xposIzq,yposIzq,xposDer,yposDer


def rotate_line_pixel5(coordCont, center, radian): 
    x = coordCont[0]
    y = coordCont[1]
    #https://stackoverflow.com/questions/13695317/rotate-a-point-around-another-point
   
    cos = np.cos(radian)
    sin = np.sin(radian)
    xposIzq = int(cos * (x - center[0]) - sin * (y - center[1]) + center[0])
    yposIzq = int(sin * (x - center[0]) + cos * (y - center[1]) + center[1])
  
    return xposIzq,yposIzq

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
    
    return vResult

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
        
    if(minXYm[1] > xy[1]):
        minXYm[1] = xy[1]   
        
    
def maxCoord(xy):
    if(maxXYm[0] < xy[0]):
        maxXYm[0] = xy[0]
        
    if(maxXYm[1] < xy[1]):
        maxXYm[1] = xy[1]   
        

def mapping(xMeters, yMeters, offsetX, offsetY):
    k = SIZE_BEAM_PIXELS/SIZE_BEAM_METERS
    return int(np.abs(k*xMeters-offsetX)),int(np.abs(k*yMeters-offsetY))

def mapping2(xMeters, yMeters, offsetX, offsetY):
    return int(np.abs((((SIZE_BEAM_PIXELS*xMeters)/SIZE_BEAM_METERS))-offsetX)),int(np.abs((((SIZE_BEAM_PIXELS*yMeters)/SIZE_BEAM_METERS))-offsetY))
    

def rotation_symmetry(x, y, point): 
    x = x - point[0]
    y = y - point[1]
    #y = -y
    x = -x
    x = np.abs(x + point[0])
    y = y + point[1]
    x, y = y, x
    return int(x),int(y)

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


def euclidian_distance(a, b):
    n1, n2, e1, e2 = a[0], b[0], a[1], b[1]
    return np.sqrt((e1 - e2)**2 + (n1 - n2)**2)

def grayToBGR(imageGray, imageBGR):
    imageOutput = cv.cvtColor(imageGray, cv.COLOR_GRAY2BGR)
    for i in range(0, imageBGR.shape[0]):
        for j in range(0, imageBGR.shape[1]):
            if(imageOutput[i,j,0] == 0): 
                continue
            else:
                for z in range(0,3):
                    imageOutput[i,j,z] = imageBGR[i,j,z]
    return imageOutput
    
        


minXYm = [-500, -500]
maxXYm = [500, 500]

ax, bx = mapeoX(-450, 450, 0, 1239)
ay, by = mapeoY(-450, 450, 0, 1239)

threadWindow = threading.Thread(target=windowUI)
threadWindow.start()


##Datos sensores


coordInit  = bufferCoord[0]

lineaPrev = None

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

playVideo = True
primera = True
listFrames = [] #Contiene cada frame
listFramesNews = [] #Contiene todos los pixeles nuevos
indexFrameBack  = 0
indexFrame = 0
indexFrameROI = 0
indexFrameY = 511
counterLines=0
last_crop_img = None

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
        lineaPixels = img[indexFrameY:indexFrameY+1, 0:col]
        lineaPixelsAdjustLight = adjust_light(lineaPixels)
        
        colorGrayLineaPixels = cv.cvtColor(lineaPixelsAdjustLight, cv.COLOR_BGR2GRAY)
        colorGrayLineaPixels = remove_water_column(colorGrayLineaPixels)
                
        xm, ym = float(bufferCoord[counterLines].x), float(bufferCoord[counterLines].y)
        xMetrosCentral, yMetrosCentral = float((xm-float(coordInit.x))), float((ym-float(coordInit.y))) 

        xPixelCentral, yPixelCentral = mapping2(xMetrosCentral, yMetrosCentral, mapa.shape[1]/2 ,mapa.shape[0]/2)
        xPixelIzquierda, yPixelIzquierda = mapping2(xMetrosCentral, yMetrosCentral+SIZE_BEAM_METERS/2,mapa.shape[1]/2 ,mapa.shape[0]/2)
        xPixelDerecha, yPixelDerecha = mapping2(xMetrosCentral, yMetrosCentral-SIZE_BEAM_METERS/2,mapa.shape[1]/2 ,mapa.shape[0]/2)

        #print(xMetrosCentral, yMetrosCentral+SIZE_BEAM_METERS/2, xMetrosCentral, yMetrosCentral-SIZE_BEAM_METERS/2)
        #print(xPixelCentral, yPixelCentral, xPixelIzquierda, yPixelIzquierda, xPixelDerecha, yPixelDerecha)
        

        xPixelCentral, yPixelCentral     =  rotation_symmetry(xPixelCentral, yPixelCentral, (mapa.shape[1]/2, mapa.shape[0]/2))
        xPixelIzquierda, yPixelIzquierda =  rotation_symmetry(xPixelIzquierda, yPixelIzquierda, (mapa.shape[1]/2, mapa.shape[0]/2))
        xPixelDerecha, yPixelDerecha     =  rotation_symmetry(xPixelDerecha, yPixelDerecha, (mapa.shape[1]/2, mapa.shape[0]/2))

        xPixelIzquierda,yPixelIzquierda =   rotate_line_pixel5((xPixelIzquierda,yPixelIzquierda),(xPixelCentral,yPixelCentral),float(bufferPose[counterLines].yaw))
        xPixelDerecha,yPixelDerecha     =   rotate_line_pixel5((xPixelDerecha,yPixelDerecha),(xPixelCentral,yPixelCentral),float(bufferPose[counterLines].yaw))
        
        

        mapa[xPixelIzquierda, yPixelIzquierda, 0:3] = 255
        mapa[xPixelCentral, yPixelCentral, 0:3] = 255
        mapa[xPixelDerecha,yPixelDerecha, 0:3] = 255 
        

        lineaPixelColor = grayToBGR(colorGrayLineaPixels, lineaPixels)
        

        puntoPintarIzq = bresenham_algorithm((xPixelCentral, yPixelCentral),(xPixelIzquierda, yPixelIzquierda))
        puntoPintarDer = bresenham_algorithm((xPixelCentral, yPixelCentral),(xPixelDerecha, yPixelDerecha))
        #bufferImages.append(Image(counterLines, lineaPixels, bufferCoord[counterLines].y, (xIzq, yIzq, xDer, yDer), widthBeam)) #Almacenamos la imagen para computar posibles intersecciones
        
        """
        if (float(bufferVelocity[counterLines].velx) > 0.25):
            height = 8
        else:
            height = 5
        """
        #CALCULO POR DISTANCIA LINEAL
        """
        if(counterLines > 0):
            distance = euclidian_distance((float(bufferCoord[counterLines-1].x), float(bufferCoord[counterLines-1].y)), (float(bufferCoord[counterLines].x), float(bufferCoord[counterLines].y)))
            height = int((distance*SIZE_BEAM_PIXELS)/SIZE_BEAM_METERS)
            if height < 1: height = 1
            print(distance)
        """  
        #CALCULO POR M/S lineal
        """
        if(counterLines > 0):
            #distance = euclidian_distance((float(bufferCoord[counterLines-1].x), float(bufferCoord[counterLines-1].y)), (float(bufferCoord[counterLines].x), float(bufferCoord[counterLines].y)))
            height = int((np.abs(float(bufferVelocity[counterLines].velx)-float(bufferVelocity[counterLines-1].velx))*SIZE_BEAM_PIXELS)/SIZE_BEAM_METERS)
            distance = euclidian_distance((float(bufferCoord[counterLines-1].x), float(bufferCoord[counterLines-1].y)), (float(bufferCoord[counterLines].x), float(bufferPose[counterLines].yaw)))
            height2 = int((distance*SIZE_BEAM_PIXELS)/SIZE_BEAM_METERS)
            if height2 > height: height = height2
            print(height, height2)
            if height < 1: height = 1
        """  

        if(counterLines > 0):
            #distance = euclidian_distance((float(bufferCoord[counterLines-1].x), float(bufferCoord[counterLines-1].y)), (float(bufferCoord[counterLines].x), float(bufferCoord[counterLines].y)))
            
            distance = euclidian_distance((float(bufferCoord[counterLines-1].x), float(bufferCoord[counterLines-1].y)), (float(bufferCoord[counterLines].x), float(bufferCoord[counterLines].y)))
            timestamp = (float(bufferCoord[counterLines].time) - float(bufferCoord[counterLines-1].time))/1000
            velocity = distance/timestamp
            height = int((velocity*SIZE_BEAM_PIXELS)/SIZE_BEAM_METERS)
            
            rotationVel = np.abs((np.abs(float(bufferPose[counterLines-1].yaw )) - np.abs(float(bufferPose[counterLines].yaw )))/timestamp)
            height2 = int((rotationVel*SIZE_BEAM_PIXELS)/SIZE_BEAM_METERS)
            #print(height2)
            if height < height2: height = height2
            #height = int(height/2)
            if height < 1: height = 1
        
        #CUANDO ROTE INTENTAR QUE RELLENE HACIA ATRAS EN VEZ DE HACIA DELANTE
        
        
        #print(bufferPose[counterLines].yaw)

        resizeLineaPixels = resize_image(lineaPixelColor, lineaPixelColor.shape[1], height)

        lineaPixelIzq = resizeLineaPixels[0:resizeLineaPixels.shape[0],0:int(resizeLineaPixels.shape[1]/2), 0:3]
        lineaPixelDer = resizeLineaPixels[0:resizeLineaPixels.shape[0],int(resizeLineaPixels.shape[1]/2):resizeLineaPixels.shape[1], 0:3]
        #print(len(puntoPintarIzq), len(puntoPintarDer), lineaPixelIzq.shape[1], lineaPixelDer.shape[1])

       
        """
        for i, pair in enumerate(puntoPintarIzq):
            if i >= lineaPixelIzq.shape[1]: break
            mapa[pair[1], pair[0]] = 255

        for i, pair in enumerate(puntoPintarDer):
            if i >= lineaPixelDer.shape[1]: break
            mapa[pair[1], pair[0]] = 255
            #while(1):
            #   k = cv.waitKey(33)
            #   if k==27:    # Esc key to stop
            #       break     
        
        """
        
        entrada = False
        i = (lineaPixelIzq.shape[1])-1
        for _, pair in enumerate(puntoPintarIzq):
            if i == 0: break
            for j in range(0,height):
                for z in range(0,3):
                    if((mapa[pair[0]+j, pair[1], z] != 0 and lineaPixelIzq[j, i, z]!=0) and np.abs(mapa[pair[0]+j, pair[1], z]  - lineaPixelIzq[j, i, z]) >=0 and np.abs(mapa[pair[0]+j, pair[1], z]  - lineaPixelIzq[j, i, z]) <= 60):
                        print(pair[1], pair[0]+j,mapa[pair[0]+j, pair[1], z], lineaPixelIzq[j, i, z], int((lineaPixelIzq[j, i, z]+mapa[ pair[0]+j, pair[1], z])/2))
                        mapa[pair[0]+j, pair[1], z] = int((lineaPixelIzq[j, i, z]+mapa[pair[0]+j, pair[1], z] )/2)
                        if z == 2:
                             mapa[pair[0]+j, pair[1], 2] = int((((mapa[pair[0]+j, pair[1], 0] )) * 0.189) + (((mapa[pair[0]+j, pair[1], 1] )) * 0.769) + (((mapa[pair[0]+j, pair[1], 2] )) * 0.393))
                             if mapa[pair[0]+j, pair[1], 2] > 255: mapa[pair[0]+j, pair[1], 2] = 255
                        elif z==1:
                             mapa[pair[0]+j, pair[1], 1] = int((((mapa[pair[0]+j, pair[1], 0]) * 0.168) + (((mapa[pair[0]+j, pair[1], 1] )) * 0.686) + (((mapa[pair[0]+j, pair[1], 2] )) * 0.349)))
                             if mapa[pair[0]+j, pair[1], 1] > 255: mapa[pair[0]+j, pair[1], 1] = 255
                        elif z==0:
                            mapa[pair[0]+j, pair[1], 0] = int((((mapa[pair[0]+j, pair[1], 0] )) * 0.131) + (((mapa[pair[0]+j, pair[1], 1] )) * 0.534) + (((mapa[pair[0]+j, pair[1], 2] )) * 0.272))
                            if mapa[pair[0]+j, pair[1], 0] > 255: mapa[pair[0]+j, pair[1], 0] = 255
                    elif lineaPixelIzq[j,i,z] != 0:
                        mapa[pair[0]+j, pair[1], z]  = lineaPixelIzq[j, i, z]
               
               
                
            i = i - 1

        for i, pair in enumerate(puntoPintarDer):
            if i >= lineaPixelDer.shape[1]-1: break
            for j in range(0,height):
                for z in range(0,3):
                    if((mapa[pair[0]+j, pair[1], z] != 0 and lineaPixelDer[j, i, z]!=0) and np.abs(mapa[pair[0]+j, pair[1], z]  - lineaPixelDer[j, i, z]) >=0 and np.abs(mapa[pair[0]+j, pair[1], z]  - lineaPixelDer[j, i, z]) <= 60):
                        mapa[pair[0]+j, pair[1], z] = int(((lineaPixelDer[j, i, z]+ mapa[pair[0]+j, pair[1], z] )/2))
                        if z == 2:
                            mapa[pair[0]+j, pair[1], 2] = int((((mapa[pair[0]+j, pair[1], 0] )) * 0.189) + (((mapa[pair[0]+j, pair[1], 1] )) * 0.769) + (((mapa[pair[0]+j, pair[1], 2] )) * 0.393))
                            if mapa[pair[0]+j, pair[1], 2] > 255: mapa[pair[0]+j, pair[1], 2] = 255
                        elif z==1:
                            mapa[pair[0]+j, pair[1], 1] = int((((mapa[pair[0]+j, pair[1], 0] )) * 0.168) + (((mapa[pair[0]+j, pair[1], 1] )) * 0.686) + (((mapa[pair[0]+j, pair[1], 2] )) * 0.349))
                            if mapa[pair[0]+j, pair[1], 1] > 255: mapa[pair[0]+j, pair[1], 1] = 255
                        elif z==0:
                            mapa[pair[0]+j, pair[1], 0] = int((((mapa[pair[0]+j, pair[1], 0] )) * 0.131) + (((mapa[pair[0]+j, pair[1], 1] )) * 0.534) + (((mapa[pair[0]+j, pair[1], 2] )) * 0.272))
                            if mapa[pair[0]+j, pair[1], 0] > 255: mapa[pair[0]+j, pair[1], 0] = 255
                    elif lineaPixelDer[j,i,z] != 0:
                        mapa[pair[0]+j, pair[1],z]  = lineaPixelDer[j,i,z]
               
                
               
         
        
        
        






        counterLines = counterLines + 1   
        indexFrameY = indexFrameY - 1
        if(indexFrameY==-1):
            indexFrameROI = indexFrameROI+1
            indexFrameY = row-1 
            
            
            
        #cv.imshow('map', mapa)
        """
        if(last_crop_img is None): 
            last_crop_img = copy.copy(lineaPixelColor)
        if(counterLines > 1):
            last_crop_img = np.concatenate((lineaPrev, last_crop_img), axis=0)
            cv.imshow('concatenate',last_crop_img)
        lineaPrev = copy.copy(lineaPixelColor)
        """
        
        window.setBackground(mapa)

        #cv.imshow('map2', mapa2)
        

  # Break the loop
  else: 
    break

#importante https://stackoverflow.com/questions/30207467/how-to-draw-3d-coordinate-axes-with-opencv-for-face-pose-estimation
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv.destroyAllWindows()
