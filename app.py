import numpy as np
import cv2
import time

from common import draw_keypoints

#file1 = '/home/diego/Dropbox/01Memoria/Data Sets/VIPeR/cam_a/000_45.bmp'
file1 = '/home/diego/Dropbox/01Memoria/Data Sets/otros/Walk1.mpg'
#file1 = '/home/diego/Videos-REID/Videos-Inchalam/Estacionamiento/14300200.avi'

'''
img = cv2.imread(file1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
'''


detector = cv2.xfeatures2d.SIFT_create()
#norm = cv2.NORM_L2
#matcher = cv2.BFMatcher(norm)

#kp = sift.detect(gray,None)
#img=cv2.drawKeypoints(gray,kp)
#kp1, desc1 = detector.detectAndCompute(frame, None)
#kp2, desc2 = detector.detectAndCompute(img2, None)

cap = cv2.VideoCapture(0)

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
i=0

while(True):
    t0 = time.clock()
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fgmask = fgbg.apply(frame) # detectar primer plano en movimiento
    retval, thresh = cv2.threshold(fgmask, 200, 256, cv2.THRESH_BINARY); #eliminar sombra
    enmask = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB); #triplicar canales, para poder compararlos con un frame en RGB
    enmask = cv2.bitwise_and(frame,enmask)
    '''
    if i%24 == 0:
        kp1, desc1 = detector.detectAndCompute(fgmask, None)
        draw_keypoints(frame,kp1)
    
        #cv2.imshow('frame2',fgmask)
    '''
    # Display the resulting frame
    #print 'frame ',i
    
        
        #t1 = time.clock() - t0
    
        #print 1. / t1
        #t0 = t1
    cv2.imshow('frame',frame)
    #cv2.imshow('fgmask',fgmask)
    cv2.imshow('thresh',thresh)
    cv2.imshow('enmask',enmask)
    i=i+1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

