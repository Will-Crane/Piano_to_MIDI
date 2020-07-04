#Key_press_detection.py
import cv2 as cv2
import numpy as np
from Median_Background_Subtraction import Median_Background_Subtraction

white_key_midi_nums = [21,23,24,26,28,29,31,33,35,36,38,40,41,43,45,47,48,50,52,53,55,57,59,60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86,88,89,91,93,95,96,98,100,101,103,105,107,108]
black_key_midi_nums = [22,25,27,30,32,34,37,39,42,44,46,49,51,54,56,58,61,63,66,68,70,73,75,78,80,82,85,87,90,92,94,97,99,102,104,106]
basic_kernel = np.ones((5,5),np.uint8)

class Key_press_detection_median:

	def __init__(self, transformed_initial_frame, key_contours, params, history=900, sample_rate=10):
		self.black_keys, self.white_keys = key_contours
		[threshold, self.min_height, self.min_area, self.top_point] = params

		self.b_model = Median_Background_Subtraction(history,threshold,sample_rate)
		self.b_model.update(transformed_initial_frame)

		#map from key centre x point to the tuple: (index within midi_num list, bool of if white key (1 for white key, 0 for black key))
		self.key_centre_to_index = {}
		index = 0
		for wk in self.white_keys:
			Cx = get_contour_centre(wk)[0]
			self.key_centre_to_index[Cx] = (index,1)
			index += 1
		index = 0
		for bk in self.black_keys:
			Cx = get_contour_centre(bk)[0]
			self.key_centre_to_index[Cx] = (index,0)
			index += 1
		
		self.midi_num_to_key_contour = {}

		#map from midi_num to key contour:
		self.midi_num_to_key_contour = {}
		for index in range(0,len(white_key_midi_nums)):
			self.midi_num_to_key_contour[white_key_midi_nums[index]] = self.white_keys[index]
		for index in range(0,len(black_key_midi_nums)):
			self.midi_num_to_key_contour[black_key_midi_nums[index]] = self.black_keys[index]

	def detect_pressed_keys(self, frame, hand_mask):
		
		pos_diff_image, neg_diff_image = self.b_model.update_and_subtract(frame)
		#get a vector which gives the range of keys that could be pressed down (inferred from position of hands)
		range_vec = self.get_valid_range_vec(hand_mask)
		#apply the hand_mask to remove the hand from the difference image
		pos_diff_image = cv2.bitwise_and(pos_diff_image, pos_diff_image, mask = hand_mask)
		neg_diff_image = cv2.bitwise_and(neg_diff_image, neg_diff_image, mask = hand_mask)

		#get contours and filter them:
		pos_contours, hierarchy = cv2.findContours(pos_diff_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		pos_contours = self.filter_contours(pos_contours,range_vec)
		neg_contours, hierarchy = cv2.findContours(neg_diff_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		neg_contours = self.filter_contours(neg_contours,range_vec)

		black_key_contours = self.get_black_key_contours(pos_contours)
		white_key_contours = self.get_white_key_contours(neg_contours)
		key_contours = black_key_contours + white_key_contours

		active_midi = self.get_corresponding_MIDI_nums(key_contours)

		""" Uncomment these lines to enable live video log
		cv2.drawContours(pos_diff_image, black_key_contours, -1, 255, thickness=1)
		cv2.drawContours(neg_diff_image, white_key_contours, -1, 255, thickness=1)
		cv2.imshow('pos',pos_diff_image)
		cv2.imshow('neg',neg_diff_image)
		cv2.waitKey(1)
		"""

		return active_midi

	def filter_contours(self, contours, range_vec):
		#filters possible key press contours:
		filtered_contours = []
		for contour in contours:
			x,y,w,h = cv2.boundingRect(contour)
			#1: valid range:
			if range_vec[0][x] == 0:
				continue
			#2: valid height:
			elif h < self.min_height:
				continue
			#3: valid top_point:
			elif y > self.top_point:
				continue
			#4: valid area:
			elif cv2.contourArea(contour) < self.min_area:
				continue
			else:
				filtered_contours.append(contour)
		return filtered_contours


	def get_black_key_contours(self, diff_contours):
		active_black_keys = []
		for contour in diff_contours:
			cnt_centre = get_contour_centre(contour)
			key_pressed = self.point_to_bk_contour(cnt_centre)
			if key_pressed is not None:
				active_black_keys.append(key_pressed)
		return active_black_keys

	def get_white_key_contours(self, diff_contours):
		active_white_keys = []
		for contour in diff_contours:
			cnt_centre = get_contour_centre(contour)
			key_pressed = self.point_to_wk_contour(cnt_centre)
			if key_pressed is not None:
				active_white_keys.append(key_pressed)
		return active_white_keys

	def point_to_bk_contour(self,point):
		"""
			Input: (x,y) point within the image
			Output: The contour within which the point lies, else if the point does not lie within key contours, None
		"""
		for contour in self.black_keys:
			#find if the point is within the contour: (pointpolygontest returns 1 if it is, else 0 or -1)
			if cv2.pointPolygonTest(contour, point, False) == 1:
				return contour
		if point[0] > 1000:
			return None
		else:
			return self.point_to_bk_contour((point[0] + 1,point[1]))

	def point_to_wk_contour(self,point):
		"""
			Input: (x,y) point within the image
			Output: The contour within which the point lies, else if the point does not lie within key contours, None
		"""
		for contour in self.white_keys:
			#find if the point is within the contour: (pointpolygontest returns 1 if it is, else 0 or -1)
			if cv2.pointPolygonTest(contour, point, False) == 1:
				return contour
		if point[0] > 1000:
			return None
		else:
			return self.point_to_wk_contour((point[0] + 1,point[1]))

	def get_corresponding_MIDI_nums(self,key_contours):
		#translate key_contour into MIDI note value:
		active_midi = set([])
		for key in key_contours:
			Cx, Cy = get_contour_centre(key)
			index, wk = self.key_centre_to_index[Cx]
			if wk:
				active_midi.add(white_key_midi_nums[index])
			else:
				active_midi.add(black_key_midi_nums[index])
		return active_midi

	def get_valid_range_vec(self,hand_mask):
		"""
			Input: binary mask of image, black everywhere but the hands which are white (255)
			Output: Lookup vector for valid X values: Usage: use x val as the index for this vector, if >0 then valid, else if =0 then not

			Method: 
			1. Reduce our mask matrix down to a vector along the x axis, using the Min function across columns.
			2. dilate this vector such that we can cover the octave/mask fail case
			   We can use this vector for lookup of any x value to see if it's in valid range of the hands
		"""
		lookup_vec = cv2.bitwise_not(cv2.reduce(hand_mask, 0, cv2.REDUCE_MIN, -1))
		lookup_vec = cv2.dilate(lookup_vec, basic_kernel, iterations=1)

		return lookup_vec

def top_two_thirds(frame):
	#returns the top two thirds of the frame
	height, width = frame.shape[0], frame.shape[1]
	crop_frame = frame[0:int(height * 2/3), 0:width].copy()
	return crop_frame

def get_contour_centre(cnt):
	x,y,w,h = cv2.boundingRect(cnt)
	Cx = x + w/2
	Cy = y + h/2
	return Cx,Cy