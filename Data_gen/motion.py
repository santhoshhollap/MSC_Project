'''
python3 motion.py <input_path> <output_path>
'''
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imutils

def calculate_2dft(input):
	ft = np.fft.fft2(input)
	return np.fft.fftshift(ft)

def calculate_2dift(input):
	ift = np.fft.ifft2(input)
	return ift

images = os.listdir(sys.argv[1])
for image in images:
	if image.endswith("motion.png"):
		image1 = cv2.imread(os.path.join(sys.argv[1],image))
		image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
		image2 = image1.copy()
		h,w = image1.shape
		midh = int(h/2)
		midw = int(w/2)
		image1 = imutils.rotate(image1, -3)
		image2 = imutils.rotate(image2, +3)
		ft1 = calculate_2dft(image1)
		ft2 = calculate_2dft(image2)
		final = np.zeros(ft1.shape,dtype="complex128")
		final[0:midh,0:midw] = ft1[0:midh,0:midw]
		final[midh:,midw:] = ft1[midh:,midw:]
		final[0:midh,midw:] = ft2[0:midh,midw:]
		final[midh:,0:midw] = ft2[midh:,0:midw]
		ift = calculate_2dift(final)
		recon = (abs(ift))
		cv2.imwrite(os.path.join(sys.argv[2],image),recon)

