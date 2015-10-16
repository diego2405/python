import os
import cv2
from app import playVideo, Par

#wd = '/home/diego/Videos-REID/micc_surveillance_dataset/run'
#wd = '/home/diego/Videos-REID/caviar/Leaving-bags-behind'
wd = '/home/diego/Videos-REID/caviar/shoping'

descriptores=[]

#playVideo(0,descriptores)

files = os.listdir(wd)
files = sorted(files)

ct = 0
for f in files:
	ct = ct + 1
	print 'descriptores almacenados:{}'.format(len(descriptores))
	if ct > 50:
		break
	nombre = '{}{}{}'.format(wd,'/',f)
	#cap = cv2.VideoCapture(nombre)
	playVideo(nombre,descriptores)
cv2.waitKey(0)
