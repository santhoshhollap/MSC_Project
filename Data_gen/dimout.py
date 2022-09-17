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

height = []
width = []

for file in files:
	if file.endswith(".png"):
		img = cv2.imread(os.path.join(sys.argv[1],file))
		(h,w,z) = img.shape
		if h <= 200 or h >= 300 or w <= 200 or w>=300:
			shutil.move(os.path.join(sys.argv[1],file), sys.argv[2])
			





			
				