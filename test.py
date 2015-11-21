import os
import cv2
import numpy as np
#from app import playVideo, Par
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

def analizar(nombre,galeria,archivo):
	global modo, minArea1, minArea2, parametros_camara
	global frame_interval, thresholdSombra, proporcionMinima, frame_interval1
	global distanciaUmbral, minMatches, frame_interval, proporcionMinima, frame_interval2
	cap = cv2.VideoCapture(nombre)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	#kernel = np.ones((3,3),np.uint8)
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	fgbg1 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
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
		fgmaskConRuido = fgbg.apply(frame) # detectar primer plano en movimiento sin sombras
		fgmaskConSombra = fgbg1.apply(frame) # detectar primer plano en movimiento con sombras
		if (i%frame_interval1) != 0:
			continue
		fgmaskSinRuido = cv2.morphologyEx(fgmaskConRuido, cv2.MORPH_OPEN, kernel) #usando kernel para eliminar ruido
		#fgmaskFiltrado = cv2.erode(fgmask,kernel)
		#retval, thresh = cv2.threshold(fgmask, thresholdSombra, 256, cv2.THRESH_BINARY); #eliminar sombra
		retval, thresh = cv2.threshold(fgmaskSinRuido, thresholdSombra, 256, cv2.THRESH_BINARY); #eliminar sombra
		mascara = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,mascara) #enmascarar frame original
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
			area = w * h
			proporcion = h / float(w)
			
			minArea = 0
			if modo == 'agregar':
				minArea = minArea1
			elif modo == 'buscar':
				minArea = minArea2

			if '180957' in archivo:
				seguir = False
				minX = parametros_camara[0]
				maxX = parametros_camara[1]
				minY = parametros_camara[2]
				maxY = parametros_camara[3]
				minArea = parametros_camara[4]
				cv2.rectangle(frame,(minX,minY),(maxX,maxY),(0,255,0),1) #dibujar rectangulo
				if (x > minX) & (x < maxX) & (y > minY) & (y < maxY) :
					seguir = True
			else:
				seguir = True

			if (area > minArea ) & (proporcion >= proporcionMinima) & seguir: #caviar-area: 500, visor-area: 1000
				#print 'proporcion',h/w
				#print 'area: ', area
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
				roi = enmask[y:y+h,x:x+w]
				roiRGB = frame[y:y+h,x:x+w]
				cv2.imshow('roi',roi)
				roi = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
				#roi = frame[y:y+h,x:x+w]
				kp, desc = detector.detectAndCompute(roi, None)
				'''
				hist = cv2.calcHist([roi],[0],None,[255],[1,255])
				#colorPrincipal = np.argmax(hist)
				#print colorPrincipal
				
				colorPrincipal = 0
				index = 0
				maxValue = 0
				#hist[0] = 0 
				for a in hist:
					#print maxValue,int(a), maxValue < int(a)
					if maxValue < int(a):
						maxValue = int(a)
						colorPrincipal = index
					index += 1 
				print colorPrincipal
				'''
				if modo == 'agregar': #MODO AGREGAR
					r = cv2.waitKey(10000)
					if r == ord('1'):
						par = Par(kp,desc,roiRGB,nombre)		
						galeria.append(par)	
						print 'en galeria:',len(galeria)
					else:
						print 'en galeria:',len(galeria)
				elif modo == 'buscar': #MODO BUSCAR
					distancia = maxint
					if len(kp)>4:
						personasDetectadas += 1
						if len(galeria) > 0:
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
								img3 = cv2.drawMatches(imagenElegida,keypointsElegidos,roiRGB,kp,matches_bajoUmbral,imagenElegida,flags=2)
								cv2.imshow('img3',img3)
								r = cv2.waitKey(10000)
								if r == ord('1'):
									TP += 1
									if TP + FP > 0:
										print 'precision: ', TP / float(TP+FP)
								elif r == ord('2'):
									FP += 1
									if TP + FP > 0:
										print 'precision: ', TP / float(TP+FP)
		cv2.imshow('original',frame)
		cv2.imshow('fgmaskConRuido',fgmaskConRuido)
		cv2.imshow('fgmaskSinRuido',fgmaskSinRuido)
		cv2.imshow('fgmaskConSombra',fgmaskConSombra)
		cv2.imshow('mascara',mascara)
		cv2.imshow('enmask',enmask)
		r = cv2.waitKey(20)
		if r == ord('q'):
			break
		elif r == ord('b'):
			modo = 'buscar'
			print modo
		elif r == ord('a'):
			modo = 'agregar'
			print modo

		#print 'frame:',i, ' rate: ', 1. / t1

	f=open('resultados.csv','a')
	if modo == 'buscar': #MODO BUSCAR
		print >>f,'tamano galeria,',len(galeria)
		precision='error'
		if TP + FP > 0:
			precision=float(TP)/float(TP+FP)
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
wd = '/home/diego/Videos-REID/caviar/shoping'
#wd = '/home/diego/Videos-REID/caviar/shopping-seleccion'
#wd = '/home/diego/Videos-REID/Panaderia'

descriptores=[]
#playVideo(0,descriptores)

distanciaUmbral = 180
minMatches = 1
frame_interval1 = 2
frame_interval2 = 2	
thresholdSombra = 200
minArea1 = 3000
minArea2 = 400 
proporcionMinima = 2.0
modo = 'agregar'

f=open('resultados.csv','a')
print >>f,'Metodo, DR'
print >>f,'Espacio colores, HSV'
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
size = len(files)
for f in files:
	print modo
	print '{} de {}'.format(i,size)
	i += 1
	nombre = '{}{}{}'.format(wd,'/',f)
	#if not 'OneStopEnter1cor' in f:
		#continue
	if 'cor.mpg' in f:
		#continue
		modo = 'agregar'
		#galeria = []  #galeria nueva para cada video
		#almacenarKeypoints(nombre,galeria)
	elif 'front.mpg' in f:
		#continue
		modo = 'buscar'
	elif '180957' in f:
		continue
		parametros_camara = [0,600,150,479,800]
		#buscarEnGaleria(nombre,galeria,f)
	analizar(nombre,galeria,f)
	#cap = cv2.VideoCapture(nombre)
	#playVideo(nombre,descriptores)
	#detectar(nombre,frame_interval,parametros_camara)
	
cv2.waitKey(0)


