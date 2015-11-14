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
	global frame_interval, thresholdSombra, proporcionMinima, frame_interval1
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
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		if (i%frame_interval1) != 0:
			continue
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) #usando kernel para eliminar ruido
		retval, thresh = cv2.threshold(fgmask, thresholdSombra, 256, cv2.THRESH_BINARY); #eliminar sombra
		enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original
		_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #mi metodo
		#_, contours0,hierarchy = cv2.findContours(fgmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # metodo paper
		contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
		# Find the index of the largest contour
		areas = [cv2.contourArea(c) for c in contours]
		if len(areas) > 0:
			#print 'ROIs:',len(areas)
			max_index = np.argmax(areas)
			cnt=contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			area = w * h
			proporcion = h / w
			if (area > minArea1 ) & (proporcion >= proporcionMinima) : #caviar-area: 500, visor-area: 1000
				#print 'proporcion',h/w
				#print 'area: ', area
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
				#roi = enmask[y:y+h,x:x+w]
				roi = frame[y:y+h,x:x+w]
				kp, desc = detector.detectAndCompute(roi, None)
				print 'keypoints: ',len(kp)
				print 'galeria size: ',len(galeria)
				r = cv2.waitKey(10000)
				if r == ord('1'):
					cv2.imshow('roi',roi)
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
	f=open('resultados.csv','a')
	print >>f,'tamano galeria',len(galeria)
	f.close()
	


def buscarEnGaleria(nombre,galeria,archivo):
	global distanciaUmbral, minMatches, frame_interval, thresholdSombra, proporcionMinima, frame_interval2
	cap = cv2.VideoCapture(nombre)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	detector = cv2.xfeatures2d.SIFT_create()
	norm = cv2.NORM_L2
	matcher = cv2.BFMatcher(norm)
	maxint = pow(2,63)-1
	i=0
	roi_index = 0
	levels=1
	TP=0
	FP=0
	personasDetectadas = 0
	while True:
		i=i+1
		ret, frame = cap.read() # Capture frame-by-frame
		if ret == False:
			break
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		if (i%frame_interval2) != 0:
			continue
		fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel) #usando kernel para eliminar ruido
		retval, thresh = cv2.threshold(fgmask, thresholdSombra, 256, cv2.THRESH_BINARY); #eliminar sombra
		enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original
		_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		#_, contours0,hierarchy = cv2.findContours(fgmask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) #metodo paper
		contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
		# Find the index of the largest contour
		areas = [cv2.contourArea(c) for c in contours]
		if len(areas) > 0:
			#print 'ROIs:',len(areas)
			max_index = np.argmax(areas)
			cnt=contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			#print 'altura:',h
			area = w * h
			proporcion = h / w
			if (area > minArea2 ) & (proporcion >= proporcionMinima): #caviar-area: 500, visor-area: 1000
				#print 'proporcion',h/w
				#print 'area: ', area
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
				#roi = enmask[y:y+h,x:x+w]
				roi = frame[y:y+h,x:x+w]
				kp, desc = detector.detectAndCompute(roi, None)
				distancia = maxint
				if len(kp)>4:
					personasDetectadas += 1
					for par in galeria:
						#raw_matches = matcher.knnMatch(par.descriptor, trainDescriptors = desc, k = 2) #2		
						matches = matcher.match(par.descriptor, desc) #este funciona
						matches = sorted(matches, key = lambda x:x.distance)
						if len(matches)>0:
							if matches[0].distance < distancia:
								distancia = matches[0].distance
								imagenElegida = par.image
								keypointsElegidos = par.keypoints
								matchesElegidos = matches
					#print 'distancia:',distancia	
					matches_bajoUmbral = []
					for m in matchesElegidos:
						if m.distance < distanciaUmbral:
							matches_bajoUmbral.append(m)
					#print 'matches_bajoUmbral:',len(matches_bajoUmbral)
					if len(matches_bajoUmbral)>=minMatches:
						img3 = cv2.drawMatches(imagenElegida,keypointsElegidos,roi,kp,matches_bajoUmbral,imagenElegida,flags=2)
						cv2.imshow('img3',img3)
						r = cv2.waitKey(10000)
						if r == ord('1'):
							TP += 1
						elif r == ord('2'):
							FP += 1
		cv2.imshow('frame',frame)
		#cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		#cv2.imshow('enmask',enmask)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
		#print 'frame:',i, ' rate: ', 1. / t1

	precision=float(TP)/float(TP+FP)
	
	f=open('resultados.csv','a')
	print >>f,'pruebas,',TP+FP
	print >>f,'TP,',TP
	print >>f,'FP,',FP
	print >>f,'precision,',precision
	print >>f,'Archivo,',archivo
	print >>f,'-------------------------------'

	f.close()
	
	cap.release()

#wd = '/home/diego/Videos-REID/micc_surveillance_dataset/run'
#wd = '/home/diego/Videos-REID/caviar/Leaving-bags-behind'
#wd = '/home/diego/Videos-REID/caviar/shoping'
wd = '/home/diego/Videos-REID/caviar/shopping-seleccion'
#wd = '/home/diego/Videos-REID/Panaderia'

descriptores=[]
#playVideo(0,descriptores)

distanciaUmbral = 250
minMatches = 1
frame_interval1 = 1
frame_interval2 = 6	
thresholdSombra = 200
minArea1 = 3000
minArea2 = 400 
proporcionMinima = 1

f=open('resultados.csv','a')
print >>f,'Metodo, DR'
print >>f,'distanciaUmbral,{}'.format(distanciaUmbral)
print >>f,'minMatches,{}'.format(minMatches)
print >>f,'frame_interval1,{}'.format(frame_interval1)
print >>f,'frame_interval2,{}'.format(frame_interval2)
print >>f,'minArea1,{}'.format(minArea1)
print >>f,'minArea2,{}'.format(minArea2)
print >>f,'proporcionMinima,{}'.format(proporcionMinima)
f.close()

files = os.listdir(wd)
files = sorted(files)


parametros_camara = [] # minX maxX minY maxY minArea
galeria = []
i = 1
for f in files:
	nombre = '{}{}{}'.format(wd,'/',f)
	if 'cor.mpg' in f:
		#print 'video',i
		i += 1
		#galeria = []  #galeria nueva para cada video
		almacenarKeypoints(nombre,galeria)
	elif 'front.mpg' in f:
		buscarEnGaleria(nombre,galeria,f)
	#cap = cv2.VideoCapture(nombre)
	#playVideo(nombre,descriptores)
	#detectar(nombre,frame_interval,parametros_camara)
	
cv2.waitKey(0)


