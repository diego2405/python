import numpy as np
import cv2

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
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    '''
    fgmask = fgbg.apply(frame)

    kp1, desc1 = detector.detectAndCompute(fgmask, None)
    draw_keypoints(frame,kp1)
    '''
    #cv2.imshow('frame2',fgmask)

    # Display the resulting frame
    print 'frame ',i
    i=i+1
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

