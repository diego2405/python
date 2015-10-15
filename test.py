import os
import cv2
from app import playVideo, Par

wd = '/home/diego/Videos-REID/micc_surveillance_dataset/car1'

files = os.listdir(wd)
files = sorted(files)

for f in files:
	nombre = '{}{}{}'.format(wd,'/',f)
	#cap = cv2.VideoCapture(nombre)
	playVideo(nombre,descriptores=[])
	