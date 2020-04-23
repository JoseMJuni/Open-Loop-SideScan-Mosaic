import pyxtf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# Leemos el archivo.

datos_file = '../recursos/datos/NBP050506A.XTF'

# Leemos el header y el packet
(header, packets) = pyxtf.xtf_read(datos_file)

# Mostramos que tiene el header
print('----Header----')
print(header)
n_channels = header.channel_count(verbose=True)
print('El numero de canales que tenemos en el sonar es: ',n_channels)


# Mostramos que tiene el packets. Aquí tendremos los pings de cada canal, necesarios para crear las imágenes.
print('----Packets----')

sonar_ch = packets[pyxtf.XTFHeaderType.sonar]  
# Cada elemento en la lista es un ping
sonar_ch_ping1 = sonar_ch[0]
print('Número de pings que tenemos: ',len(sonar_ch))

# The properties in the header defines the attributes common for all subchannels 
# (e.g sonar often has port/stbd subchannels)
print('Mostramos lo que hay en el ping 1: \n',sonar_ch_ping1)

# The data and header for each subchannel is contained in the data and ping_chan_headers respectively.
# The data is a list of numpy arrays (one for each subchannel)
sonar_subchan0 = sonar_ch_ping1.data[0]  # type: np.ndarray
sonar_subchan1 = sonar_ch_ping1.data[1]  # type: np.ndarray

print(sonar_subchan0.shape)
print(sonar_subchan1.shape)


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
ax1.semilogy(np.arange(0, sonar_subchan0.shape[0]), sonar_subchan0)
ax2.semilogy(np.arange(0, sonar_subchan1.shape[0]), sonar_subchan1)

sonar_ping1_canal1 = np.array(sonar_subchan0)
sonar_ping1_canal1 = np.interp(sonar_ping1_canal1, (sonar_ping1_canal1.min(), sonar_ping1_canal1.max()), (0, 255))

prueba = np.array([[sonar_ping1_canal1], [len(sonar_ping1_canal1)]])
print(prueba.shape)
#cv.imshow('valor',np.random.uniform(0,255,(100,100)))
fig, (ax1) = plt.subplots(1,1, figsize=(12,8))
#ax1.imshow(np.random.uniform(0,255,(100,100)), cmap='gray')
ax1.imshow(np.random.uniform(0,255,(100,100)), cmap='gray')

array = np.random.randint(255, size=(200,300,1),dtype=np.uint8)
print(array.shape)
cv.imshow('RGB',array)
cv.waitKey(0)