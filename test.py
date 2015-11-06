import os
import cv2
import numpy as np
from app import playVideo, Par
from detectar_movimiento import detectar
from common import draw_keypoints, anorm, getsize
from find_obj import explore_match, filter_matches

class Par():
	def __init__(self,keypoints,descriptor,image,nombreVideo):
		self.keypoints = keypoints
		self.descriptor = descriptor
		self.image = image
		self.nombreVideo = nombreVideo


def almacenarKeypoints(nombre,galeria):
	cap = cv2.VideoCapture(nombre)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	detector = cv2.xfeatures2d.SIFT_create()
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
		if (i%24) != 0:
			continue
		#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) #usando kernel para eliminar ruido
		#retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
		#enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		#enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original
		#_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #mi metodo
		_, contours0,hierarchy = cv2.findContours(fgmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # metodo paper
		contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
		# Find the index of the largest contour
		areas = [cv2.contourArea(c) for c in contours]
		if len(areas) > 0:
			#print 'ROIs:',len(areas)
			max_index = np.argmax(areas)
			cnt=contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			#area = w * h
			#print 'area: ', area
			#if (area > 2000 ) :#& (w<h) : #caviar-area: 500, visor-area: 1000
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
			#roi = enmask[y:y+h,x:x+w]
			roi = frame[y:y+h,x:x+w]
			kp, desc = detector.detectAndCompute(roi, None)
			#print len(kp)
			par = Par(kp,desc,roi,nombre)	
			galeria.append(par)


		cv2.imshow('frame',frame)
		#cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		#cv2.imshow('enmask',enmask)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
		#print 'frame:',i, ' rate: ', 1. / t1
	cap.release()
	print len(galeria)


def buscarEnGaleria(nombre,galeria,archivo):
	cap = cv2.VideoCapture(nombre)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	detector = cv2.xfeatures2d.SIFT_create()
	norm = cv2.NORM_L2
	matcher = cv2.BFMatcher(norm)
	i=0
	roi_index = 0
	levels=1
	TP=0
	FP=0
	while True:
		i=i+1
		ret, frame = cap.read() # Capture frame-by-frame
		if ret == False:
			break
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		if (i%24) != 0:
			continue
		#fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) #usando kernel para eliminar ruido
		#retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
		#enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		#enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original
		#_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		_, contours0,hierarchy = cv2.findContours(fgmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #metodo paper
		contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
		# Find the index of the largest contour
		areas = [cv2.contourArea(c) for c in contours]
		if len(areas) > 0:
			#print 'ROIs:',len(areas)
			max_index = np.argmax(areas)
			cnt=contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			#print 'altura:',h
			#area = w * h
			#print 'area: ', area
			#if (area > 100 ) & (h > w): #caviar-area: 500, visor-area: 1000
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
			#roi = enmask[y:y+h,x:x+w]
			roi = frame[y:y+h,x:x+w]
			kp, desc = detector.detectAndCompute(roi, None)
			if len(kp)>4:
				for par in galeria:
					#raw_matches = matcher.knnMatch(par.descriptor, trainDescriptors = desc, k = 2) #2		
					matches = matcher.match(par.descriptor, desc) 		
					matches = sorted(matches, key = lambda x:x.distance)
					matchesMask = [[0,0] for i in xrange(len(matches))]
					draw_params = dict(matchColor = (0,255,0),
						singlePointColor = (255,0,0),
						matchesMask = matchesMask,
						flags = 0)
					
					#if len(matches)>0:
					#	print 'menor:',matches[0].distance
					
					matches_bajoUmbral = []
					for m in matches:
						if m.distance < 300:
							matches_bajoUmbral.append(m)
					#print 'matches_bajoUmbral:',len(matches_bajoUmbral)
					if len(matches_bajoUmbral)>4:
						img3 = cv2.drawMatches(par.image,par.keypoints,roi,kp,matches_bajoUmbral,par.image,0)
						cv2.imshow('img3',img3)
						r = cv2.waitKey(10000)
						if r == ord('1'):
							TP += 1
							print 'TP'
						elif r == ord('2'):
							FP += 1
							print 'FP'
		cv2.imshow('frame',frame)
		#cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		#cv2.imshow('enmask',enmask)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
		#print 'frame:',i, ' rate: ', 1. / t1

	#precision=TP/(TP+FP)
	print '{},{},{}'.format(archivo,TP,FP)
	f=open('resultados.csv','a')
	print >>f,'{},{},{}'.format(archivo,TP,FP)
	f.close()

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
galeria = []
for f in files:
	nombre = '{}{}{}'.format(wd,'/',f)
	if 'cor.mpg' in f:
		galeria = []
		almacenarKeypoints(nombre,galeria)
	elif 'front.mpg' in f:
		buscarEnGaleria(nombre,galeria,f)
		'''
		print 'Salir? (s/n)'
		r=raw_input()
		if r=='s':
			break
		else:
			continue
		'''
	#cap = cv2.VideoCapture(nombre)
	#playVideo(nombre,descriptores)
	#detectar(nombre,frame_interval,parametros_camara)
	
cv2.waitKey(0)


