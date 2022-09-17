'''
python3 view.py <input_path> <output_path>
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
	if image.endswith("view.png"):
		image1 = cv2.imread(os.path.join(sys.argv[1],image))
		image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
		image2 = np.zeros((256,256))
		h,w = image2.shape
		num1 = random.randint(0, 3)
		area = random.randint(0, 20)

		if num1 == 0: #crop left
			image2[:,0:w-area] = image1[:,area:]
		elif num1 == 1: #crop bottom
			image2[area:,:] = image1[0:h-area,:]
		elif num1 == 2: #crop right
			image2[:,area:] = image1[:,0:w-area]
		elif num1 == 3: #crop top
			image2[0:h-area,:] = image1[area:,:]

		cv2.imwrite(os.path.join(sys.argv[2],image),image2)
