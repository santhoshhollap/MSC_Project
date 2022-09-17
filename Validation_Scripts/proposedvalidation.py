import cv2
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imutils
import shutil


def createMaskforGT(GTmask,IoU,Dice):
	img0 = np.zeros((256,256))
	img1 = np.zeros((256,256))
	img2 = np.zeros((256,256))
	img3 = np.zeros((256,256))
	if IoU:
		img0[np.where(GTmask==0)] = 255
		img1[np.where(GTmask==1)] = 255
		img2[np.where(GTmask==2)] = 255
		img3[np.where(GTmask==3)] = 255
	else:
		img0[np.where(GTmask==0)] = 1
		img1[np.where(GTmask==1)] = 1
		img2[np.where(GTmask==2)] = 1
		img3[np.where(GTmask==3)] = 1
	return img0,img1,img2,img3



def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union


def calculateIOU(target,prediction):
	'''
	This function is responsible for calculating the IOU for given masks
	Input : Segmented mask, groundtruth mask
	output : IOU value
	'''
	intersection = np.logical_and(target, prediction)
	union = np.logical_or(target, prediction)
	iou_score = np.sum(intersection) / np.sum(union)
	return iou_score

def getPred(pred,IoU,Dice):
	img0 = np.zeros((256,256))
	img1 = np.zeros((256,256))
	img2 = np.zeros((256,256))
	img3 = np.zeros((256,256))
	img0copy = np.zeros((256,256))

	img0[np.where(pred==0)] = 0
	img0[np.where(pred>0)] = 1
	img0[np.where(pred>120)] = 2
	img0[np.where(pred>200)] = 3


	if Dice:
		img0copy[np.where(img0==0)] = 1
		img1[np.where(img0==1)] = 1
		img2[np.where(img0==2)] = 1
		img3[np.where(img0==3)] = 1
	else:
		img0copy[np.where(img0==0)] = 255
		img1[np.where(img0==1)] = 255
		img2[np.where(img0==2)] = 255
		img3[np.where(img0==3)] = 255

	tempimg = np.ones((256,256))
	tempimg[0:20,:] = 0
	tempimg[236:,:] = 0
	tempimg[:,0:20] = 0
	tempimg[:,236:] = 0
	img1 = img1 * tempimg
	img0copy[0:20,:] = 1
	img0copy[236:,:] = 1
	img0copy[:,0:20] = 1
	img0copy[:,236:] = 1

	return img0copy,img1,img2,img3


IoU = False
Dice = True

count0 = 0
count1 = 0
count2 = 0
count3 = 0

dice0 = []
dice1 = []
dice2 = []
dice3 = []

files = os.listdir(sys.argv[1])
for file in files:
	if file.endswith(".png"):
		# print("working on file ",file)
		pred = cv2.imread(os.path.join(sys.argv[1],file))
		maskfile = file.split("_fake")[0] + "_mask.png"
		print(maskfile)
		GTmask = cv2.imread(os.path.join(sys.argv[2],maskfile))
		GTmask = cv2.cvtColor(GTmask,cv2.COLOR_BGR2GRAY)
		pred = cv2.cvtColor(pred,cv2.COLOR_BGR2GRAY)

		GTm0,GTm1,GTm2,GTm3 =  createMaskforGT(GTmask,IoU,Dice)
		pred0,pred1,pred2,pred3	= getPred(pred,IoU,Dice)

		if IoU:
			if len(np.where(GTm0==255)[0]) > 0:
				dice0.append(calculateIOU(np.asarray(GTm0),np.asarray(pred0)))
				count0 = count0 + 1
			if len(np.where(GTm1==255)[0]) > 0:
				dice1.append(calculateIOU(np.asarray(GTm1),np.asarray(pred1)))
				count1 = count1 + 1
			if len(np.where(GTm2==255)[0]) > 0:
				dice2.append(calculateIOU(np.asarray(GTm2),np.asarray(pred2)))
				count2 = count2 + 1
			if len(np.where(GTm3==255)[0]) > 0:
				dice3.append(calculateIOU(np.asarray(GTm3),np.asarray(pred3)))
				count3 = count3 + 1
		if Dice:
			if len(np.where(GTm0==1)[0]) > 0:
				dice0.append(dice_metric(np.asarray(GTm0),np.asarray(pred0)))
				count0 = count0 + 1
			if len(np.where(GTm1==1)[0]) > 0:
				dice1.append(dice_metric(np.asarray(GTm1),np.asarray(pred1)))
				count1 = count1 + 1
			if len(np.where(GTm2==1)[0]) > 0:
				dice2.append(dice_metric(np.asarray(GTm2),np.asarray(pred2)))
				count2 = count2 + 1
			if len(np.where(GTm3==1)[0]) > 0:
				dice3.append(dice_metric(np.asarray(GTm3),np.asarray(pred3)))
				count3 = count3 + 1

if IoU:
	print("average IOU score for 0",np.sum(dice0)/count0)
	print("average IOU score for 1",np.sum(dice1)/count1)
	print("average IOU score for 2",np.sum(dice2)/count2)
	print("average IOU score for 3",np.sum(dice3)/count3)
	print("overall IOU",((np.sum(dice0)/count0) + (np.sum(dice1)/count1) + (np.sum(dice2)/count2) + (np.sum(dice3)/count3)) / 4 )

if Dice:
	print("average Dice score for 0",np.sum(dice0)/count0)
	print("average Dice score for 1",np.sum(dice1)/count1)
	print("average Dice score for 2",np.sum(dice2)/count2)
	print("average Dice score for 3",np.sum(dice3)/count3)
	print("overall Dice",((np.sum(dice0)/count0) + (np.sum(dice1)/count1) + (np.sum(dice2)/count2) + (np.sum(dice3)/count3)) / 4 )



		