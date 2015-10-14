import os
import cv2
from app import playVideo


wd = '/home/diego/Videos-REID/micc_surveillance_dataset/car1'

files = os.listdir(wd)
files = sorted(files)
for f in files:
	nombre = '{}{}{}'.format(wd,'/',f)
	cap = cv2.VideoCapture(nombre)
	if cap.isOpened():
		while True:
			ret, frame = cap.read() # Capture frame-by-frame
			if ret == False:
				break
			cv2.imshow('frame',frame)
			if cv2.waitKey(50) & 0xFF == ord('w'):
				break
			#print f
		if cv2.waitKey(50) & 0xFF == ord('q'):
				break
	#cv2.waitKey(50)
	