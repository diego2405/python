import numpy as np
import cv2
import time

from common import draw_keypoints, anorm, getsize
from find_obj import explore_match, filter_matches
from matplotlib import pyplot as plt

dirsalida = '/home/diego/salida/'
i = 0
def indiceAreaMaxima(areas):
	max_index = np.argmax(areas)
	areas[max_index] = 0.0
	return max_index

def detectar(nombre,speed,parametros_camara):
	cap = cv2.VideoCapture(nombre)
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	global i
	i=0
	roi_index = 0
	levels=1
	ruta = []
	while True:
		i = i + 1
		t0 = time.clock() # calcular tiempo
		ret, frame = cap.read() # Capture frame-by-frame
		if ret == False:
			break
		if (i % speed) != 0:
			continue
		#if i < 2700 | (i > 4000 & i < 7200 ):
		#	continue
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
		enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original

		#enmask = cv2.cvtColor(enmask, cv2.COLOR_RGB2GRAY)#convertir mascara a uint8
		#hist_mask = cv2.calcHist([frame],[0],enmask,[256],[0,256]) #calcular histograma
		#plotHistograma(frame,enmask)
		_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # encontrar contornos
		contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0] #encontrar polinomio aproximado para contornos
		areas = [cv2.contourArea(c) for c in contours] # Find the index of the largest contour

		rectangulos=dibujarContornosEntreRangos(frame,contours,parametros_camara,ruta)
		for r in rectangulos:
			guardarROI(frame,r[0],r[1],r[2],r[3])
		cv2.putText(frame,"frame: {}".format(i),(10,30),cv2.FONT_HERSHEY_DUPLEX,1, (0,255,0))
		dibujarRuta(frame,ruta)

		cv2.imshow('frame',frame)
		#cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		cv2.imshow('enmask',enmask)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
		t1 = time.clock() - t0
		t0 = t1
		#print 'frame:',i, ' rate: ', 1. / t1
		
	#cap.release()


def dibujarContornosEntreRangos(frame,contours,parametros_camara,ruta):
	minX = parametros_camara[0]
	maxX = parametros_camara[1]
	minY = parametros_camara[2]
	maxY = parametros_camara[3]
	minArea = parametros_camara[4]
	#cv2.rectangle(frame,(minX,minY),(maxX,maxY),(0,255,0),1) #dibujar rectangulo del area de interes
	rectangulos = []
	for c in contours:
		if cv2.contourArea(c) < minArea :
			continue
		x,y,w,h = cv2.boundingRect(c)
		if w > h:
			continue
		if (x > minX) & (x < maxX) & (y > minY) & (y < maxY):
			rectangulos.append((x,y,w,h))
			#cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
			#cv2.putText(frame,"({})".format(cv2.contourArea(c)),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0))
			ruta.append((int(x+(w/2)), int(y+(h/2))))
	return rectangulos

def dibujarRuta(frame,ruta):
	for punto in ruta:
		cv2.circle(frame, (punto[0],punto[1]), 2, (0,255,0)) #dibujar punto

def plotHistograma(frame,mascara):
	color = ('b','g','r')
	for i,col in enumerate(color):
		histr = cv2.calcHist([frame],[i],mascara,[256],[0,256])
		plt.plot(histr,color = col)
		plt.xlim([0,256])
		plt.show()

def guardarROI(frame,x,y,w,h):
	global dirsalida
	roi = frame[y:y+h,x:x+w]
	cv2.imwrite('{}cf_{}.png'.format(dirsalida ,i),roi)
	'''
	if len(areas) > 0:
			#print 'ROIs:',len(areas)
			#max_index = np.argmax(areas)
			#print 'indice area:', max_index
			max_index = indiceAreaMaxima(areas)
			cnt=contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			area = w * h
			if (area > 100 ) : #caviar-area: 500, visor-area: 1000
				cv2.putText(frame,"({},{})".format(x,y),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0))
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
				#cv2.circle(frame, (int(x+(w/2)), int(y+(h/2))), 2, (0,255,0)) #dibujar punto en centro del rectangulo
				#roi_index = roi_index + 1
				#roi = enmask[y:y+h,x:x+w]

			max_index = indiceAreaMaxima(areas)
			cnt = contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			area = w * h
			if (area > 100 ): #caviar-area: 500, visor-area: 1000
				cv2.putText(frame,"loc: ({},{})".format(x,y),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0))
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
				#cv2.circle(frame, (int(x+(w/2)), int(y+(h/2))), 2, (0,255,0)) #dibujar punto en centro del rectangulo
	'''

