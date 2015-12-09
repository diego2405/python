import os
import cv2
import numpy as np
#from app import playVideo, Par
from detectar_movimiento import detectar
from common import draw_keypoints, anorm, getsize
from find_obj import explore_match, filter_matches
from matplotlib import pyplot as plt

class Tupla():
	def __init__(self,keypoints,descriptor,imageRGB,imageHSV,nombreVideo):
		self.keypoints = keypoints
		self.descriptor = descriptor
		self.imageRGB = imageRGB
		self.imageHSV = imageHSV
		self.nombreVideo = nombreVideo


def analizar(nombre,galeria,archivo):
	global modo, minArea1, minArea2, parametros_camara, usoMascara, norma
	global frame_interval, thresholdSombra, proporcionMinima, frame_interval1
	global distanciaUmbral, minMatches, frame_interval, proporcionMinima, frame_interval2
	cap = cv2.VideoCapture(nombre)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	#kernel = np.ones((3,3),np.uint8)
	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	fgbg1 = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
	detector = cv2.xfeatures2d.SIFT_create()
	norm = cv2.NORM_L1
	#matcher = cv2.BFMatcher(norm)
	matcher = cv2.BFMatcher(norm, True)
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
		conRectangulo = frame.copy()
		conPoligono = frame.copy()
		conTodo = frame.copy()
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
		#areas = [cv2.contourArea(c) for c in contours]
		#if len(areas) > 0:
		maxArea = 0
		cnt = None
		#cv2.drawContours(frame,contours,-1,(0,255,0))
		con = []
		
		for c in contours:
			x1,y1,w1,h1 = cv2.boundingRect(c)
			#cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),1) #dibujar rectangulo
			proporcion = h1 / float(w1)
			if proporcion > proporcionMinima:
				if cv2.contourArea(c) > maxArea:
					x,y,w,h = cv2.boundingRect(c)
					maxArea = cv2.contourArea(c)
					area = w * h
					cnt = c
					con = []
					con.append(c)
					proporcion = h / float(w)
		if not cnt == None:
			minArea = 0
			if modo == 'agregar':
				minArea = minArea1
			elif modo == 'buscar':
				minArea = minArea1
			if '180957' in archivo:
				seguir = False
				minX = parametros_camara[0]
				maxX = parametros_camara[1]
				minY = parametros_camara[2]
				maxY = parametros_camara[3]
				minArea = parametros_camara[4]
				#cv2.rectangle(frame,(minX,minY),(maxX,maxY),(0,255,0),1) #dibujar rectangulo
				if (x > minX) & (x < maxX) & (y > minY) & (y < maxY) :
					seguir = True
			else:
				seguir = True

			if (area > minArea ) & (proporcion >= proporcionMinima) & seguir: 
				if usoMascara:
					roiRGB = enmask[y:y+h,x:x+w]
				else:
					roiRGB = frame[y:y+h,x:x+w]
				cv2.imshow('roiRGB',roiRGB)
				roiHSV = cv2.cvtColor(roiRGB,cv2.COLOR_BGR2HSV)
				#cv2.imshow('roiHSV',roiHSV)
				if modo == 'agregar': #MODO AGREGAR
					r = cv2.waitKey(10000)
					if r == ord('1'):
						if espacioColores == 'RGB':
							kp, desc = detector.detectAndCompute(roiRGB, None)
						elif espacioColores == 'HSV':
							kp, desc = detector.detectAndCompute(roiHSV, None)
						tupla = Tupla(kp,desc,roiRGB,roiHSV,nombre)		
						galeria.append(tupla)	
						print 'en galeria:',len(galeria)
				elif modo == 'buscar': #MODO BUSCAR
					if espacioColores == 'RGB':
							kp, desc = detector.detectAndCompute(roiRGB, None)
					elif espacioColores == 'HSV':
							kp, desc = detector.detectAndCompute(roiHSV, None)
					ii = 1
					ss = len(galeria)
					for t in galeria:
						if ii < 2:
							resized_image = cv2.resize(t.imageRGB, (50, 200)) 
							img1 = resized_image.copy()
						else:
							resized_image = cv2.resize(t.imageRGB, (50, 200)) 
							img2 = resized_image.copy()
							img1 = np.concatenate((img1, img2), axis=1)
						#plt.subplot(int('{}{}{}'.format(1,ss,ii))),plt.imshow(t.imageRGB,'gray')
						ii += 1
					cv2.imshow('galeria',img1)
					distancia = maxint
					if len(kp)>4:
						personasDetectadas += 1
						if len(galeria) > 0:
							matchesElegidos = None
							for t in galeria:
								#matches = matcher.knnMatch(t.descriptor, desc, k = 2) #2		
								matches = matcher.match(t.descriptor, desc) #este funciona
								#print len(matches)
								matches = sorted(matches, key = lambda x:x.distance)
								if len(matches) >= 10:
									tempDist = 0
									for m in matches[:10]:
										tempDist += m.distance
									#print 'distancia promedio',tempDist,distancia
									if tempDist < distancia:
										distancia = tempDist
										imagenElegida = t.imageRGB
										keypointsElegidos = t.keypoints
										matchesElegidos = matches

								'''
								# Need to draw only good matches, so create a mask
								matchesMask = [[0,0] for i in xrange(len(matches))]
								for i,(m,n) in enumerate(matches):
									if m.distance < 0.7 * n.distance:
										matchesMask[i]=[1,0]
										if m.distance < distancia:
											distancia = m.distance
											imagenElegida = t.imageRGB
											keypointsElegidos = t.keypoints
											matchesElegidos = matches

								'''
							'''
							matches_bajoUmbral = []
							for m in matchesElegidos:
								if m.distance < distanciaUmbral:
									matches_bajoUmbral.append(m)
							#print 'matches_bajoUmbral:',len(matches_bajoUmbral)
							'''
							#if len(matches_bajoUmbral)>=minMatches:
							if not matchesElegidos == None:
								'''
								draw_params = dict(matchColor = (0,255,0),
								                   singlePointColor = (255,0,0),
								                   matchesMask = matchesMask,
								                   flags = 0)
								img3 = cv2.drawMatchesKnn(imagenElegida,keypointsElegidos,roiRGB,kp,matchesElegidos,None,**draw_params)
								'''
								img3 = cv2.drawMatches(
									imagenElegida,
									keypointsElegidos,
									roiRGB,
									kp,
									#matches_bajoUmbral,
									matchesElegidos,  
									imagenElegida,
									flags=2)
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
		#cv2.imshow('fgmaskConRuido',fgmaskConRuido)
		#cv2.imshow('fgmaskSinRuido',fgmaskSinRuido)
		#cv2.imshow('fgmaskConSombra',fgmaskConSombra)
		#cv2.imshow('conRectangulo',conRectangulo)
		#cv2.imshow('conPoligono',conPoligono)
		#cv2.imshow('conTodo',conTodo)
		#cv2.imshow('mascara',mascara)
		#cv2.imshow('enmask',enmask)
		r = cv2.waitKey(20)
		if r == ord('q'):
			break
		elif r == ord('b'):
			modo = 'buscar'
			print modo
		elif r == ord('a'):
			modo = 'agregar'
			print modo

		cv2.destroyWindow('img3')
		cv2.destroyWindow('roiRGB')
	

	f=open('resultados.csv','a')
	precision='error'
	if TP + FP > 0:
		precision=float(TP)/float(TP+FP)
	#######  ar pr tp fp ts ga dM pM aM co ma mo no
	linea = '{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
		archivo,
		precision,
		TP,
		FP,
		TP+FP,
		len(galeria),
		distanciaUmbral,
		proporcionMinima,
		minArea1,
		espacioColores,
		usoMascara,
		modo,
		norma)
	print >>f, linea
	f.close()

	cap.release()


#wd = '/home/diego/Videos-REID/micc_surveillance_dataset/run'
#wd = '/home/diego/Videos-REID/caviar/Leaving-bags-behind'
wd = '/home/diego/Videos-REID/caviar/shoping'
#wd = '/home/diego/Videos-REID/caviar/shopping-seleccion'
#wd = '/home/diego/Videos-REID/Panaderia'

descriptores=[]
#playVideo(0,descriptores)


minMatches = 1
frame_interval1 = 2
frame_interval2 = 2	
thresholdSombra = 200
minArea1 = 3000
minArea2 = 400 
proporcionMinima = 2.0
modo = 'agregar'
espacioColores = 'RGB'
usoMascara = False
norma = 'L1'

if norma == 'L1':
	distanciaUmbral = 1200
elif norma == 'L2':
	distanciaUmbral = 180


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
		continue
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


