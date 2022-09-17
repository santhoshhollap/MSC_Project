'''
python3 missing.py <input_path> <output_path>
'''
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imutils
import random

images = os.listdir(sys.argv[1])
for image in images:
	if image.endswith("missing.png"):
		image1 = cv2.imread(os.path.join(sys.argv[1],image))
		image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
		image1[110:120,110:120] = 0
		image1[80:90,100:110] = 0
		image1[40:50,200:210] = 0
		cv2.imwrite(os.path.join(sys.argv[2],image),image1)

