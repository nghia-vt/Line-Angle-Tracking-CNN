"""
Create dataset of line image.
Format of output image name: <video-name>_<frame-index>_<line-angle>
	video-name: name of video to be captured
	frame-index: Index of frame in the video
	line-angle: angle (deg) of line in the image
Usage:
	Run 'python3 labeling_captured_sample.py --help' to see all argument in detail
	Example: 'python3 labeling_captured_sample.py -vid video1.avi -per 0.1 -pth dataset/ -type png'
"""

import argparse
import numpy as np
import math
from cv2 import cv2
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("-vid", "--videofile", type=str, help="video will be captured")
ap.add_argument("-pth", "--folderpath", type=str, default="dataset/", help="folder to store samples (default = dataset/")
ap.add_argument("-per", "--period", type=float, default=1/24, help="time interval of capturing sample in second (default = 1/24)")
ap.add_argument("-type", "--imgtype", type=str, default="png", help="type of image to be saved (default = png)")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["videofile"])
if (cap.isOpened() == False):
    print("Could not open the Video")

# Print status
print("** Capturing & Labling samples **")
print("   Folder containing sample data: {0}/{1}".format(os.getcwd(), args["folderpath"]))
print("   Press key 'q' to stop")

frameRate = cap.get(cv2.CAP_PROP_FPS) # frame per second
frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frame of video
frameIndex = 0

def calc_steer_angle(image):
	h, w = image.shape[0], image.shape[1]
	roi = image[round(h/2):h, 0:w]

	gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	edges_roi = cv2.Canny(gray_roi, 50, 75, apertureSize=3)

	lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, 100, minLineLength=80, maxLineGap=40)
	if lines is None:
		return None
	
	angles = []
	for line in lines:
		x1,y1,x2,y2 = line[0]
		if (y1 - y2) == 0:
			theta = 90
		else:
			theta = math.atan((x2 - x1)/(y1 - y2))
			theta = 90 - np.rad2deg(theta) # counter-clockwise is positive
		angles.append(theta)
		# cv2.line(roi, (x1,y1), (x2,y2), (0,255,0), 2) # comment this line while saving samples
	theta_avg = round(np.mean(angles))
	return theta_avg

while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Labeling the frame (format: <video-name>_<frame-index>_<line-angle>)
	angle = calc_steer_angle(frame)
	# print(angle)
	if angle is not None:
		folder_path = args["folderpath"]
		img_name = args["videofile"].split(".")[0] + '_' + str(frameIndex) + '_' + str(angle)
		img_extension = '.' + args["imgtype"]
		cv2.imwrite(folder_path + img_name + img_extension, frame)

	# Display the resulting frame
	cv2.imshow('Capturing Sample', frame)
	# time.sleep(1/frameRate)

	# Index of the next frame to be captured
	frameIndex += round(frameRate*args["period"])
	cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)

	# Print progress status
	sys.stdout.write("\r\tProgress: {0}% ... ".format(round(100*frameIndex/frameCount, 1)))
	sys.stdout.flush()

	# Press Q on keyboard to stop capture
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

print("\tCompleted")
cap.release()
cv2.destroyAllWindows()