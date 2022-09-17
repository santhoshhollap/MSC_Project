'''
python3 ghosting_motion.py <input_path> <output_path>
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
	if image.endswith("motionGhosting.png"):
		image1 = cv2.imread(os.path.join(sys.argv[1],image))
		image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
		h,w = image1.shape
		dimw = int(w/2)
		dimh = int(h/2)
		image2 = np.zeros(image1.shape,dtype="uint8")
		image2[0:h-10,:] = image1[10:,:]
		image3 = np.zeros(image1.shape,dtype="uint8")
		image3[10:,:] = image1[0:h-10,:]
		img3 = cv2.addWeighted(src1=image1[:,0:dimw], alpha=0.85, src2=image2[:,0:dimw], beta=0.15, gamma=0.0)
		img3 = cv2.addWeighted(src1=img3, alpha=0.85, src2=image3[:,0:dimw], beta=0.15, gamma=0.0)
		image1[:,0:dimw] = img3

		image2 = image1.copy()
		h,w = image1.shape
		midh = int(h/2)
		midw = int(w/2)
		image1 = imutils.rotate(image1, -1)
		image2 = imutils.rotate(image2, +1)
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

