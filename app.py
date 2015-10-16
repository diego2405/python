import numpy as np
import cv2
import time

from common import draw_keypoints, anorm, getsize
from find_obj import explore_match, filter_matches

dirsalida = '/home/diego/salida/'
class Par():
	def __init__(self,keypoints,descriptor,image):
		self.keypoints = keypoints
		self.descriptor = descriptor
		self.image = image

def playVideo(nombre, descriptores):
	detector = cv2.xfeatures2d.SIFT_create()
	norm = cv2.NORM_L2
	matcher = cv2.BFMatcher(norm)

	cap = cv2.VideoCapture(nombre)

	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	i=0
	roi_index = 0
	levels=1
	while True:
		t0 = time.clock() # calcular tiempo
		ret, frame = cap.read() # Capture frame-by-frame
		if ret == False:
			break
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
		enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original
		_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
		# Find the index of the largest contour
		areas = [cv2.contourArea(c) for c in contours]
		if len(areas) > 0:
			print 'ROIs:',len(areas)
			max_index = np.argmax(areas)
			cnt=contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			area = w * h
			print 'area: ', area
			if (area > 100 ) & (h > w): #caviar-area: 500, visor-area: 1000
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1) #dibujar rectangulo
				cv2.circle(frame, (int(x+(w/2)), int(y+(h/2))), 2, (0,255,0)) #dibujar punto en centro del rectangulo
				roi_index = roi_index + 1
				roi = enmask[y:y+h,x:x+w]
				kp, desc = detector.detectAndCompute(roi, None)
				print len(kp)
				#print 'posicion: ({},{}) tamano:({},{}) area: {}'.format(x,y,w,h,area)
				if len(kp) > 0: #caviar: 4 , visor:20
					for par in descriptores:
						raw_matches = matcher.knnMatch(par.descriptor, trainDescriptors = desc, k = 2) #2		
						p1, p2, kp_pairs = filter_matches(par.keypoints, kp, raw_matches)
						if len(p1) >= 4:
							H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
							if status is not None:
								print '%d / %d  inliers/matched' % (np.sum(status), len(status))
								'''
								tmp1 = 'match{}_1.png'.format(i)
								tmp2 = 'match{}_2.png'.format(i)
								cv2.imwrite('{}{}'.format(dirsalida,tmp1),par.image)
								cv2.imwrite('{}{}'.format(dirsalida,tmp2),roi)
								'''
							vis = explore_match('find_obj', roi, par.image, kp_pairs, status, H)
						else:
							H, status = None, None
							#print '%d matches found, not enough for homography estimation' % len(p1)
					par = Par(kp,desc,roi)	
					print 'descriptores guardados:', len(descriptores)
					descriptores.append(par)
		cv2.imshow('frame',frame)
		#cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		#cv2.imshow('enmask',enmask)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
		t1 = time.clock() - t0
		t0 = t1
		#print 'frame:',i, ' rate: ', 1. / t1
		i=i+1
	#cap.release()

