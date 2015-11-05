import numpy as np
import cv2
import time

from common import draw_keypoints, anorm, getsize
from find_obj import explore_match, filter_matches

dirsalida = '/home/diego/salida/'

def segmentar(nombre):
	cap = cv2.VideoCapture(nombre)

	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	i=0
	roi_index = 0
	levels=1
	while True:

		ret, frame = cap.read() # Capture frame-by-frame
		if ret == False:
			break
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
		enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original

		cv2.imshow('frame',frame)
		cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		#cv2.imshow('enmask',enmask)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
		#print 'frame:',i, ' rate: ', 1. / t1
		i=i+1
	#cap.release()



