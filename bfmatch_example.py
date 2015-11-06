import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('box.png',0)          # queryImage
img2 = cv2.imread('box_in_scene.png',0) # trainImage

detector = cv2.xfeatures2d.SIFT_create()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)

orb = cv2.ORB_create()

#kp1, des1 = orb.detectAndCompute(img1,None)
#kp2, des2 = orb.detectAndCompute(img2,None)

kp1, des1 = detector.detectAndCompute(img1,None)
kp2, des2 = detector.detectAndCompute(img2,None)


#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
bf = cv2.BFMatcher(norm)

matches = bf.match(des1,des2)

matches = sorted(matches, key = lambda x:x.distance)
i = 0
matches_bajoUmbral = []
for m in matches:
	if m.distance < 100:
		matches_bajoUmbral.append(m)

print 'matches_bajoUmbral:',len(matches_bajoUmbral)

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img1)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches_bajoUmbral, img1)

cv2.imshow('salida',img3)

cv2.waitKey(0)