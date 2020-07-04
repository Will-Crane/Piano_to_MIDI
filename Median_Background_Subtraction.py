#Median_Background_Subtraction.py
import numpy as np
import cv2 as cv2
import math

class Median_Background_Subtraction:
	def __init__(self,history,threshold,sample_rate):
		#history: size of time window for which we store previous frames (in #frames)
		#threshold: threshold value for difference images
		#sample_rate: how many of the frames within the time window do we actually consider (used to speed up method). 1 = sample every frame, 10 = sample every 10 frames
		self.history = history
		self.threshold = threshold
		self.sample_rate = sample_rate

		self.frame_queue = None
		self.frame_index = 0
		self.frame_counter = 0
		self.queue_size = int(self.history / self.sample_rate)

		self.median_background = None

	def update_and_subtract(self,frame):
		self.update(frame)
		return self.subtract(frame)

	def update(self,frame):
		#work in greyscale
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		if self.frame_queue is None:
			#initialise the queue and the background model:
			self.frame_queue = np.array([frame] * self.queue_size)
			self.median_background = frame.copy()
			self.frame_index += 1
		elif self.frame_counter < self.sample_rate:
			#skip this frame for efficiency reasons:
			self.frame_counter += 1
		else:
			#add this frame into our queue and update the background model
			self.frame_counter = 0
			self.frame_queue[self.frame_index] = frame
			self.median_background = np.median(self.frame_queue,axis=0).astype(np.uint8)
			if self.frame_index == self.queue_size - 1:
				self.frame_index = 0
			else:
				self.frame_index += 1

	def subtract(self,frame):
		#returns positive and negative difference images (thresholded)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		pos_diff = cv2.subtract(frame,self.median_background)
		neg_diff = cv2.subtract(self.median_background,frame)
		return self.take_thresholds(pos_diff, neg_diff)

	def combine(self,pos_diff,neg_diff):
		#combines positive and negative difference images into a single difference image
		return pos_diff + neg_diff

	def take_threshold(self,combined_diff_image):
		#thresholds the positive and negative difference images
		ret, thresholded = cv2.threshold(combined_diff_image,self.threshold,255,cv2.THRESH_BINARY)
		return thresholded

	def take_thresholds(self,pos_diff,neg_diff):
		ret, pos_thresh = cv2.threshold(pos_diff,self.threshold,255,cv2.THRESH_BINARY)
		ret, neg_thresh = cv2.threshold(neg_diff,self.threshold,255,cv2.THRESH_BINARY)
		return pos_thresh, neg_thresh

	def get_median_background_image(self):
		return self.median_background.copy()