import os
import cv2
import sys
import time
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
	global modo, minArea, maxArea, parametros_camara, usoMascara, norma
	global frame_interval, thresholdSombra, proporcionMinima, proporcionMaxima, frame_interval1
	global distanciaUmbral, numeroMinimoMatches, porcentajeMinimoMatches, frame_interval, salidaGaleriaNueva
	#cap = cv2.VideoCapture(0)
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
	
	#cargar galeria inicial para caviar
	galeria = []
	nombreGaleria = '/home/diego/salida/galeriaPanaderia/exp1/personas'
	if modo == 'buscar':
		filesGaleria = os.listdir(nombreGaleria)
		i = 1
		for f in filesGaleria:
			if (('subida' in f) & ('subida' in archivo)) | (('bajada' in f) & ('bajada' in archivo)):
				continue 
			imageRGB = cv2.imread('{}/{}'.format(nombreGaleria,f),cv2.IMREAD_COLOR)
			imageHSV = cv2.cvtColor(imageRGB,cv2.COLOR_BGR2HSV)
			if espacioColores == 'RGB':
				kp, desc = detector.detectAndCompute(imageRGB, None)
			elif espacioColores == 'HSV':
				kp, desc = detector.detectAndCompute(imageHSV, None)
			tupla = Tupla(kp,desc,imageRGB,imageHSV,f)		
			galeria.append(tupla)
			i += 1	
	
	i=0
	roi_index = 0
	TP=0
	FP=0
	countTiempoPromedioMatch = 0
	tiempoPromedioMatch = 0
	personasDetectadas = 0
	promedioSumaDistanciaMinimaCorrecta = 0
	promedioSumaDistanciaMinimaIncorrecta = 0
	while True:
		i=i+1

		ret, frame = cap.read() # Capture frame-by-frame
		if ret == False:
			break
		#cv2.imshow('stream',frame)
		conRectangulo = frame.copy()
		conPoligono = frame.copy()
		conTodo = frame.copy()
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmaskConRuido = fgbg.apply(frame) # detectar primer plano en movimiento sin sombras
		fgmaskConSombra = fgbg1.apply(frame) # detectar primer plano en movimiento con sombras
		if ((i+1)%frame_interval1) != 0:
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
		maxAreaTemp = 0
		cnt = None
		#cv2.drawContours(frame,contours,-1,(0,255,0))
		
		for c in contours:
			x1,y1,w1,h1 = cv2.boundingRect(c)
			#cv2.rectangle(frame,(x1,y1),(x1+w1,y1+h1),(255,0,0),1) #dibujar rectangulo
			'''
			if '180957' in archivo:
				minX = parametros_camara[0]
				maxX = parametros_camara[1]
				minY = parametros_camara[2]
				maxY = parametros_camara[3]
				minArea = parametros_camara[4]
				#cv2.rectangle(frame,(minX,minY),(maxX,maxY),(0,255,0),1) #dibujar rectangulo
				if (x1 < minX) | (x1 > maxX) | (y1 < minY) | (y1 > maxY) :
					continue
			'''
			tempProporcion = h1 / float(w1)
			tempArea = w1 * h1
			if (tempArea > minArea) & (tempArea < maxArea):
				#if tempProporcion > proporcionMaxima:
				if tempProporcion > proporcionMinima:
					if cv2.contourArea(c) > maxAreaTemp:
						x,y,w,h = cv2.boundingRect(c)
						maxAreaTemp = cv2.contourArea(c)
						area = w * h
						#print 'area',area, ' tempArea:',tempArea, ' minArea:',minArea
						cnt = c
						proporcion = tempProporcion
		if cnt != None:
			if usoMascara:
				roiRGB = enmask[y:y+h,x:x+w]
			else:
				roiRGB = frame[y:y+h,x:x+w]
			roiHSV = cv2.cvtColor(roiRGB,cv2.COLOR_BGR2HSV)
			#cv2.imshow('roiRGB',roiRGB)
			#cv2.imshow('roiHSV',roiHSV)
			if modo == 'agregar': #MODO AGREGAR
				#r = cv2.waitKey(10000)
				#if r == ord('1'):
				personasDetectadas += 1
				if espacioColores == 'RGB':
					kp, desc = detector.detectAndCompute(roiRGB, None)
				elif espacioColores == 'HSV':
					kp, desc = detector.detectAndCompute(roiHSV, None)
				if len(kp) >= 10:
					tupla = Tupla(kp,desc,roiRGB,roiHSV,nombre)
					tempNombre = '{}/{}_{:03}.png'.format(salidaGaleriaNueva,archivo,personasDetectadas)
					#print tempNombre
					cv2.imwrite(tempNombre, roiRGB)	
					galeria.append(tupla)	
				#	print 'en galeria:',len(galeria)
				'''
				elif r == ord('n'):
					break
				elif r == ord('q'):
					sys.exit()
				elif r == ord('b'):
					modo = 'buscar'
					print modo
				elif r == ord('a'):
					modo = 'agregar'
					print modo
				'''
			elif modo == 'buscar': #MODO BUSCAR
				if espacioColores == 'RGB':
						kp, desc = detector.detectAndCompute(roiRGB, None)
				elif espacioColores == 'HSV':
						kp, desc = detector.detectAndCompute(roiHSV, None)
				''' 
				#mostrar galeria
				ii = 1
				ss = len(galeria)
				for t in galeria:
					if ii < 2:
						r = 200.0 / t.imageRGB.shape[0]
						dim = (int(t.imageRGB.shape[1] * r),200)
						resized_image = cv2.resize(t.imageRGB, dim, interpolation = cv2.INTER_AREA)
						img1 = resized_image.copy()
					else:
						r = 200.0 / t.imageRGB.shape[0]
						dim = (int(t.imageRGB.shape[1] * r),200)
						resized_image = cv2.resize(t.imageRGB, dim, interpolation = cv2.INTER_AREA)
						img2 = resized_image.copy()
						img1 = np.concatenate((img1, img2), axis=1)
					#kp, desc = detector.detectAndCompute(t.imageRGB, None)
					#cv2.imwrite('{}_{}.png'.format(ii,len(kp)),t.imageRGB)
					ii += 1
				'''
				if len(galeria) > 0:	
					#cv2.imshow('galeria',img1)
					distancia = maxint
					if len(kp)>=10:
						matchesElegidos = None
						for t in galeria:
							tempMinimoMatches = numeroMinimoMatches
							#matches = matcher.knnMatch(t.descriptor, desc, k = 2) #2	
							t0 = time.clock()	
							matches = matcher.match(t.descriptor, desc) #este funciona
							t1 = time.clock()
							tiempoPromedioMatch += (t1 - t0)
							countTiempoPromedioMatch += 1
							#print len(matches)
							matches = sorted(matches, key = lambda x:x.distance)
							#numeroMinimoMatches = int(len(matches) * porcentajeMinimoMatches)
							
							if tempMinimoMatches > len(matches):
								tempMinimoMatches = len(matches)
							#print tempMinimoMatches, len(matches)
							if len(matches) >= tempMinimoMatches & tempMinimoMatches > 0:
								tempDist = 0
								for m in matches[:tempMinimoMatches]:
									tempDist += m.distance
								tempDist /= tempMinimoMatches
								if tempDist < distancia:
									distancia = tempDist
									imagenElegida = t.imageRGB
									nombreElegido = t.nombreVideo
									keypointsElegidos = t.keypoints
									matchesElegidos = matches
						#print '---'
						if matchesElegidos != None:
							'''
							draw_params = dict(matchColor = (0,255,0),
							                   singlePointColor = (255,0,0),
							                   matchesMask = matchesMask,
							                   flags = 0)
							img3 = cv2.drawMatchesKnn(imagenElegida,keypointsElegidos,roiRGB,kp,matchesElegidos,None,**draw_params)
							'''
							r = 200.0 / roiRGB.shape[0]
							dim = (int(roiRGB.shape[1] * r),200)
							img1 = cv2.resize(roiRGB, dim, interpolation = cv2.INTER_AREA)

							r = 200.0 / imagenElegida.shape[0]
							dim = (int(imagenElegida.shape[1] * r),200)
							img2 = cv2.resize(imagenElegida, dim, interpolation = cv2.INTER_AREA)

							img1 = np.concatenate((img1, img2), axis=1)
							#cv2.imshow('match',img1)
							#print '{}\t{}\t{}'.format(len(keypointsElegidos),len(kp),len(matchesElegidos))
							cv2.imwrite('{}/{}_{:03}.png'.format(salidaGaleriaNueva,archivo,i),img1)
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
							cv2.waitKey(0)
							
							r = cv2.waitKey(10000)
							if r == ord('1'):
								TP += 1
								if tempMinimoMatches > 0:
									promedioSumaDistanciaMinimaCorrecta += distancia/tempMinimoMatches
								if TP + FP > 0:
									print 'precision: ', TP / float(TP+FP)
							elif r == ord('2'):
								FP += 1
								if tempMinimoMatches > 0:
									promedioSumaDistanciaMinimaIncorrecta += distancia/tempMinimoMatches
								if TP + FP > 0:
									print 'precision: ', TP / float(TP+FP)
							elif r == ord('n'):
								break
							elif r == ord('q'):
								sys.exit()
							elif r == ord('b'):
								modo = 'buscar'
								print modo
							elif r == ord('a'):
								modo = 'agregar'
								print modo
							'''
									
		#cv2.imshow('stream',frame)
		#cv2.imshow('fgmaskConRuido',fgmaskConRuido)
		#cv2.imshow('fgmaskSinRuido',fgmaskSinRuido)
		#cv2.imshow('fgmaskConSombra',fgmaskConSombra)
		#cv2.imshow('conRectangulo',conRectangulo)
		#cv2.imshow('conPoligono',conPoligono)
		#cv2.imshow('conTodo',conTodo)
		#cv2.imshow('mascara',mascara)
		#cv2.imshow('enmask',enmask)
		'''
		r = cv2.waitKey(20)
		if r == ord('n'):
			break
		elif r == ord('q'):
			sys.exit()
		elif r == ord('b'):
			modo = 'buscar'
				modo
		elif r == ord('a'):
			modo = 'agregar'
			print modo
		elif r == ord('s'):
			print 'guardando'
			f=open('resultados.csv','a')
			precision='error'
			if TP + FP > 0:
				precision=float(TP)/float(TP+FP)
				if TP > 0:
					promedioSumaDistanciaMinimaCorrecta /= TP
				if FP > 0:
					promedioSumaDistanciaMinimaIncorrecta /= FP
			
			if countTiempoPromedioMatch > 0:
				tiempoPromedioMatch /= countTiempoPromedioMatch
			#######  ar pr tp fp ts ga mM dM pM aM co ma no mc mi tm pm
			linea = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
				archivo, #0
				precision, #1
				TP,#2
				FP,#3
				TP+FP,#4
				len(galeria),#5
				distanciaUmbral,#6
				proporcionMinima,#7
				minArea,#8
				espacioColores,#9
				usoMascara,#10
				norma,#11
				promedioSumaDistanciaMinimaCorrecta,#12
				promedioSumaDistanciaMinimaIncorrecta,#13
				numeroMinimoMatches,#14
				tiempoPromedioMatch, #15
				porcentajeMinimoMatches) #16
			if not precision == 'error':
				print >>f, linea
			f.close()
			precision='error'
			TP = 0
			FP = 0
			promedioSumaDistanciaMinimaCorrecta = 0
			promedioSumaDistanciaMinimaIncorrecta = 0
			tiempoPromedioMatch = 0
			countTiempoPromedioMatch = 0

		cv2.destroyWindow('img3')
		cv2.destroyWindow('roiRGB')
		'''
	

	f=open('resultados.csv','a')
	precision='error'
	if TP + FP > 0:
		precision=float(TP)/float(TP+FP)
		if TP > 0:
			promedioSumaDistanciaMinimaCorrecta /= TP
		if FP > 0:
			promedioSumaDistanciaMinimaIncorrecta /= FP
	
	if countTiempoPromedioMatch > 0:
		tiempoPromedioMatch /= countTiempoPromedioMatch
	#######  ar pr tp fp ts ga mM dM pM aM co ma no mc mi tm
	linea = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}'.format(
		archivo, #0
		precision, #1
		TP,#2
		FP,#3
		TP+FP,#4
		len(galeria),#5
		distanciaUmbral,#6
		proporcionMinima,#7
		minArea,#8
		espacioColores,#9
		usoMascara,#10
		norma,#11
		promedioSumaDistanciaMinimaCorrecta,#12
		promedioSumaDistanciaMinimaIncorrecta,#13
		numeroMinimoMatches,#14
		tiempoPromedioMatch) #15
	if not precision == 'error':
		print >>f, linea
	f.close()

	cap.release()

'''
# load the image and show it
image = cv2.imread("box.png")
cv2.imshow("original", image)
cv2.waitKey(0)

print image.shape
'''

#wd = '/home/diego/Videos-REID/micc_surveillance_dataset/run'
#wd = '/home/diego/Videos-REID/caviar/Leaving-bags-behind'
#wd = '/home/diego/Videos-REID/caviar/shoping'
#wd = '/home/diego/Videos-REID/caviar/shopping-seleccion'
#wd = '/home/diego/Videos-REID/Panaderia'
wd = '/home/diego/camara1'

descriptores=[]
#playVideo(0,descriptores)

porcentajeMinimoMatches = 0.05
numeroMinimoMatches = 8
frame_interval1 = 1
thresholdSombra = 200
minArea = 3000
maxArea = 13000
proporcionMinima = 1.125
modo = 'buscar'
#modo = 'agregar'
espacioColores = 'HSV'
usoMascara = False
norma = 'L2'

if norma == 'L1':
	distanciaUmbral = 1200
elif norma == 'L2':
	distanciaUmbral = 180

salidaGaleriaNueva = '/home/diego/salida/galeriaPanaderia/exp9'
files = os.listdir(wd)
'''
files = {
"EnterExitCrossingPaths1cor.mpg",
"EnterExitCrossingPaths2cor.mpg",
"OneLeaveShopReenter1cor.mpg",
"OneShopOneWait1cor.mpg",
"OneShopOneWait2cor.mpg",
"OneStopEnter1cor.mpg",
"OneStopEnter2cor.mpg",
"OneStopMoveEnter2cor.mpg",
"OneStopMoveNoEnter1cor.mpg",
#"ShopAssistant1cor.mpg",
#"ShopAssistant2cor.mpg",
"WalkByShop1cor.mpg"
}
'''
files = sorted(files)



parametros_camara = [0,928,0,500,400] # minX maxX minY maxY minArea
#parametros_camara = [0,600,150,479,800]
galeria = []
#analizar('/home/diego/Videos-REID/Panaderia/ch01-151022-180957-190926.avi',galeria,'ch01-151022-180957-190926.avi')
#analizar('/home/diego/Videos-REID/Panaderia/seq2.wmv',galeria,'seq2.wmv')
#sys.exit()
i = 0
size = len(files)
inicio = time.time()
for f in files:
	i += 1
	print 'video: {}'.format(f)
	
	nombre = '{}{}{}'.format(wd,'/',f)
	#if not 'OneStopEnter1cor' in f:
		#continue
	#if 'cor.mpg' in f:
		#continue
		#modo = 'agregar'
		#galeria = []  #galeria nueva para cada video
		#almacenarKeypoints(nombre,galeria)
	if 'front.mpg' in f:
		continue
		modo = 'buscar'
	elif '180957' in f:
		continue
		parametros_camara = [0,600,150,479,800]
		#buscarEnGaleria(nombre,galeria,f)
	elif 'reid' in f:
		continue

	analizar(nombre,galeria,f)
	#cap = cv2.VideoCapture(nombre)
	#playVideo(nombre,descriptores)
	#detectar(nombre,frame_interval,parametros_camara)
	

print 'segundos transcurridos',time.time() - inicio
cv2.waitKey(0)
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
