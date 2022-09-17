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
		name = file.split("gt")[0] + file.split("gt")[1].split("_")[1] + "_" +file.split("gt")[1].split("_")[2] 
		print(file,name)
		cv2.imwrite(os.path.join(sys.argv[2],name),img)




			
				