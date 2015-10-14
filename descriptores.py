import cv2
import numpy as np
import find_objf
from find_obj import explore_match, filter_matches
from common import draw_keypoints

file1 = '/home/diego/Videos-REID/descriptores/personaAzulWalk2/descriptor8.png'
file2 = '/home/diego/Videos-REID/descriptores/personaAzulWalk2/descriptor12.png'

detector = cv2.xfeatures2d.SIFT_create()
norm = cv2.NORM_L2
matcher = cv2.BFMatcher(norm)

img1 = cv2.imread(file1,0)
img2 = cv2.imread(file2,0)

kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

draw_keypoints(img1,kp1)
draw_keypoints(img2,kp2)


print 'matching...'
raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

if len(p1) >= 4:
	H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
	print '%d / %d  inliers/matched' % (np.sum(status), len(status))
else:
	H, status = None, None
	print '%d matches found, not enough for homography estimation' % len(p1)

print len(p1), len(p2)
print p1, p2
vis = explore_match('find_obj', img2, img1, kp_pairs, status, H)



#cv2.imshow('img1',img1)
#cv2.imshow('img2',img2)


cv2.waitKey()
cv2.destroyAllWindows()