"""
Usage:
	Run 'python3 labeling_captured_sample.py --help' to see all argument in detail
	Example: 'python3 record_video.py -cam 0 -fn video1'
  (the video will be saved in '.avi' format)
"""

import numpy as np
from cv2 import cv2
import os
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cam", "--camera", type = int, default = 0, help = "index of camera")
ap.add_argument("-fn", "--filename", type = str, default = 'unknown_video', help = "filename of video will record")
args = vars(ap.parse_args())


cap = cv2.VideoCapture(args["camera"])
if (cap.isOpened() == False):
  print("Could not open camera")

frameSize = (640, 480)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameSize[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameSize[1])
fps = 24
filename = args["filename"]

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(filename+'.avi', fourcc, fps, frameSize)

print("Press Q on keyboard to stop recording.\nRecording...")

while (True):
  ret, frame = cap.read()
  if not ret:
    break

  # Write the frame into the file 'filename.avi'
  out.write(frame)

  # Display the resulting frame
  cv2.imshow('Recording Video',frame)

  # Press Q on keyboard to stop recording
  if cv2.waitKey(1) & 0xFF == ord('q'):
    print("("+filename+'.avi)'+' is saved at '+os.getcwd())
    break
    
# When everything done, release the video capture and video write objects
cap.release()
out.release()
cv2.destroyAllWindows()