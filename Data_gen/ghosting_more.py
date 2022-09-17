'''
python3 ghosting_more.py <input_path> <output_more_ghosting_path>
'''
import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

images = os.listdir(sys.argv[1])
for image in images:
	if image.endswith("moreghosting.png"):
		image1 = cv2.imread(os.path.join(sys.argv[1],image))
		image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
		image2 = np.zeros(image1.shape,dtype="uint8")
		image3 = np.zeros(image1.shape,dtype="uint8")
		h,w = image1.shape
		dimh = int(h/2)
		image2[0:dimh,:] = image1[dimh:,:]
		image3[dimh:,:] = image1[0:dimh,:]
		img3 = cv2.addWeighted(src1=image1, alpha=0.80, src2=image2, beta=0.20, gamma=0.0)
		img3 = cv2.addWeighted(src1=img3, alpha=0.80, src2=image3, beta=0.20, gamma=0.0)
		cv2.imwrite(os.path.join(sys.argv[2],image),img3)
