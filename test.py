import os
import cv2
import numpy as np
from app import playVideo, Par
from detectar_movimiento import detectar

def segmentar(nombre):
	cap = cv2.VideoCapture(nombre)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	#fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
	i=0
	roi_index = 0
	levels=1
	while True:
		i=i+1
		ret, frame = cap.read() # Capture frame-by-frame
		if ret == False:
			break
		if False:#(i%24) != 0:
			continue
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) #usando kernel para eliminar ruido
		retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
		enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original

		#cv2.imshow('frame',frame)
		#cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		cv2.imshow('enmask',enmask)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
		#print 'frame:',i, ' rate: ', 1. / t1
	cap.release()

#wd = '/home/diego/Videos-REID/micc_surveillance_dataset/run'
#wd = '/home/diego/Videos-REID/caviar/Leaving-bags-behind'
wd = '/home/diego/Videos-REID/caviar/shoping'
#wd = '/home/diego/Videos-REID/Panaderia'

descriptores=[]

#playVideo(0,descriptores)

files = os.listdir(wd)
files = sorted(files)

frame_interval = 1
parametros_camara = [] # minX maxX minY maxY minArea
for f in files:
	nombre = '{}{}{}'.format(wd,'/',f)
	#print nombre
	if f[3] == '1': #camara 1
		parametros_camara = [0,600,150,479,800]
		if f[12] != '1': # video de la tarde (tercero)
			continue
		segmentar(nombre)	
	elif f[3] == '2': #camara 2
		parametros_camara = [90,580,90,479,900]
		continue
	elif f[0] == 'P':
		parametros_camara = [0,600,150,479,800]
	#print 'descriptores almacenados:{}'.format(len(descriptores))
	#if ct != 6:
	#	continue
	
	#cap = cv2.VideoCapture(nombre)
	#playVideo(nombre,descriptores)
	#detectar(nombre,frame_interval,parametros_camara)
	
cv2.waitKey(0)

def hacerMosaico(directorio):
	directorio="/home/diego/salida/{}".format(directorio)
	archivos = os.listdir(directorio)
	for f in archivos:
		cv2.imread('{}{}'.format(directorio,f))
		cv2.imshow(f,img)		


