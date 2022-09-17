import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imutils
import nibabel as nib 
import gzip
import shutil

direc = os.listdir(sys.argv[1])


for file in direc:
	if file.endswith(".nii"):
		filepath = os.path.join(sys.argv[1],file)
		# print("working for ",file)
		img = nib.load(filepath)
		img_fdata = img.get_fdata()
		(x,y,z)=img.shape
		print(x,y,z)
		count = 0
		for i in range(z):
			count = count + 1
			silce = img_fdata[:, :, i]
			if file.endswith("gt.nii"):
				silce[np.where(silce==0)] = 0
				silce[np.where(silce==1)] = 100
				silce[np.where(silce==2)] = 180
				silce[np.where(silce==3)] = 255
				name = file.split(".")[0] + "_" + str(count) + ".png"			
				cv2.imwrite(os.path.join(sys.argv[3],name),silce)
			else:
				name = file.split(".")[0] + "_" + str(count) + ".png"			
				cv2.imwrite(os.path.join(sys.argv[2],name),silce)


			
				