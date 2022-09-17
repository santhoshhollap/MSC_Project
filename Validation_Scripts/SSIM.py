'''
python3 SSIM.py <input_path_GT> <Input_Output>
'''


import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import imutils


folder = os.listdir(sys.argv[1])

score = 0
score1 = 0
count = 0

for file in folder:
	if file.endswith(".png"):
		img1 = cv2.imread(os.path.join(sys.argv[1],file))
		name = file.split(".")[0] + "_fake_B.png"
		img2 = cv2.imread(os.path.join(sys.argv[2],name))
		grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
		grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
		score = score + ssim(grayA, grayB)
		count = count+1
		
		print("pix2pix SSIM: {}".format(ssim(grayA, grayB)))

print("overall pix2pix ssim ",score/count)