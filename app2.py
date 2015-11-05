import cv2
import os
import numpy as np
from common import draw_keypoints, anorm, getsize
from find_obj import explore_match, filter_matches

def hacerMosaico(directorio):
	directorio="/home/diego/salida/{}".format(directorio)
	archivos = os.listdir(directorio)
	
	img1 = cv2.imread('{}/{}'.format(directorio,archivos[0]),1)
	img2 = cv2.imread('{}/{}'.format(directorio,archivos[1]),1)
	img3 = cv2.imread('{}/{}'.format(directorio,archivos[2]),1)

	h1, w1 = img1.shape[:2]
	h2, w2 = img2.shape[:2]

	#create empty matrix
	vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

	#combine 2 images
	vis[:h1, :w1,:3] = img1
	vis[:h2, w1:w1+w2,:3] = img2

	detector = cv2.xfeatures2d.SIFT_create()
	norm = cv2.NORM_L2
	matcher = cv2.BFMatcher(norm)

	#vis = np.concatenate((img1, img2), axis=1)

	#cv2.imshow('MOSAICO',vis)	

	kp1, desc1 = detector.detectAndCompute(vis, None)
	kp2, desc2 = detector.detectAndCompute(img3, None)

	raw_matches = matcher.knnMatch(desc2, trainDescriptors = desc1, k = 2) #2		
	p1, p2, kp_pairs = filter_matches(kp2, kp1, raw_matches)
	print len(p1)
	if len(p1) >= 1:
		H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
		if status is not None:
			print '%d / %d  inliers/matched' % (np.sum(status), len(status))
			vis = explore_match('find_obj', vis, img3, kp_pairs, status, H)
	else:
		H, status = None, None

	cv2.waitKey(0)

hacerMosaico('p1')