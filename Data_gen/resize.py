import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imutils
import nibabel as nib 
import gzip
import shutil

files = os.listdir(sys.argv[1])

for file in files:
	if file.endswith(".png"):
		print("working on file ", file)
		img = cv2.imread(os.path.join(sys.argv[1],file))
		img = cv2.resize(img,(256,256))
		cv2.imwrite(os.path.join(sys.argv[2],file),img)




			
				