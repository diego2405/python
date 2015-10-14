import numpy as np
import cv2
import time

from common import draw_keypoints

#file1 = '/home/diego/Dropbox/01Memoria/Data Sets/VIPeR/cam_a/000_45.bmp'
#file1 = '/home/diego/Dropbox/01Memoria/Data Sets/otros/Walk1.mpg'
#file1 = '/home/diego/Videos-REID/Videos-Inchalam/Estacionamiento/14300200.avi'
file1 = '/home/diego/Videos-REID/micc_surveillance_dataset/car1_0.avi'

detector = cv2.xfeatures2d.SIFT_create()
#norm = cv2.NORM_L2
#matcher = cv2.BFMatcher(norm)

#kp = sift.detect(gray,None)
#img=cv2.drawKeypoints(gray,kp)
#kp1, desc1 = detector.detectAndCompute(frame, None)
#kp2, desc2 = detector.detectAndCompute(img2, None)

cap = cv2.VideoCapture(file1)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
i=0
levels=1
while(True):
	t0 = time.clock() # calcular tiempo
	ret, frame = cap.read() # Capture frame-by-frame
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convertir frame a escala de grises
	fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
	retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
	enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
	enmask = cv2.bitwise_and(frame,enmask) #enmascarar frame original
	_, contours0,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = [cv2.approxPolyDP(cnt, 3, True) for cnt in contours0]
	#cv2.drawContours( frame, contours, (-1, 3)[levels <= 0], (128,255,255),1, cv2.LINE_AA, hierarchy, abs(levels) )
    #    cv2.imshow('contours', vis)
	'''
    if i%24 == 0:
    	kp1, desc1 = detector.detectAndCompute(fgmask, None)
    	draw_keypoints(frame,kp1)
    	#cv2.imshow('frame2',fgmask)

	'''
	
	# Find the index of the largest contour
	areas = [cv2.contourArea(c) for c in contours]
	if len(areas) > 0:
		max_index = np.argmax(areas)
		cnt=contours[max_index]
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
		area = w * h
		#print 'area: ', w*h
		if area > 1000:
			kp1, desc1 = detector.detectAndCompute(enmask, None)
			draw_keypoints(frame,kp1)
			#cv2.imshow('frame2',f)
	
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
	if cv2.waitKey(1) & 0xFF == ord('q'):
		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()
		break

