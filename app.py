import numpy as np
import cv2
import os
import time

from common import draw_keypoints, anorm, getsize
from find_obj import explore_match, filter_matches

#file1 = '/home/diego/Dropbox/01Memoria/Data Sets/VIPeR/cam_a/000_45.bmp'
file1 = '/home/diego/Dropbox/01Memoria/Data Sets/otros/Walk1.mpg'
file2 = '/home/diego/Dropbox/01Memoria/Data Sets/otros/Walk2.mpg'
file3 = '/home/diego/Dropbox/01Memoria/Data Sets/otros/Walk3.mpg'


#file1 = '/home/diego/Videos-REID/Videos-Inchalam/Estacionamiento/14300200.avi'
file5 = '/home/diego/Videos-REID/micc_surveillance_dataset/car1_0.avi'

def playVideo(nombre):
	file4 = '/home/diego/Videos-REID/descriptores/personaAzulWalk1/descriptor43.png' #galeria inicial
	detector = cv2.xfeatures2d.SIFT_create()
	norm = cv2.NORM_L2
	matcher = cv2.BFMatcher(norm)

	# descriptor para galeria inicial
	img2 = cv2.imread(file4,0)
	kp2, desc2 = detector.detectAndCompute(img2, None)


	#kp = sift.detect(gray,None)
	#img=cv2.drawKeypoints(gray,kp)
	#kp1, desc1 = detector.detectAndCompute(frame, None)
	#kp2, desc2 = detector.detectAndCompute(img2, None)

	cap = cv2.VideoCapture(nombre)

	fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
	i=0
	roi_index = 0
	levels=1
	while True:
		t0 = time.clock() # calcular tiempo
		ret, frame = cap.read() # Capture frame-by-frame
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
		fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
		retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
		enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
		enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original
		_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
		contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
		#cv2.drawContours( frame, contours, (-1, 3)[levels <= 0], (128,255,255),1, cv2.LINE_AA, hierarchy, abs(levels) )
		#    cv2.imshow('contours', vis)
	
		
		# Find the index of the largest contour
		areas = [cv2.contourArea(c) for c in contours]
		if len(areas) > 0:
			
			max_index = np.argmax(areas)
			cnt=contours[max_index]
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			area = w * h
			#print 'area: ', w*h
			if area > 1000 :
				roi_index = roi_index + 1
				#nombre = '/home/diego/Videos-REID/descriptores/descriptor{}.png'.format(roi_index)
				#print nombre
				roi = enmask[y:y+h,x:x+w]
				#cv2.imwrite(nombre,roi)
				#roi = cv2.imread(nombre,0)
				kp1, desc1 = detector.detectAndCompute(roi, None)
				raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
				p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
				if len(p1) >= 4:
					H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
					print '%d / %d  inliers/matched' % (np.sum(status), len(status))
					#vis = explore_match('find_obj', img2, roi, kp_pairs, status, H)
				else:
					H, status = None, None
					#print '%d matches found, not enough for homography estimation' % len(p1)
				#draw_keypoints(roi,kp1)
				#cv2.imshow('ROI',roi)
				#vis = explore_match('find_obj', img2, roi, kp_pairs, status, H)
		
		# Display the resulting frame
		#cv2.imshow('fgmask',fgmask)
		#cv2.imshow('thresh',thresh)
		#cv2.imshow('enmask',enmask)
		#cv2.imshow('gray',gray)
		cv2.imshow('frame',frame)
		t1 = time.clock() - t0
		t0 = t1
		#print 'frame:',i, ' rate: ', 1. / t1
		i=i+1
		if cv2.waitKey(50) & 0xFF == ord('q'):
			# When everything done, release the capture
			break
	#cap.release()

