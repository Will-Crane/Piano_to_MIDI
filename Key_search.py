#Key_search_v2
import numpy as np
import cv2 as cv2
import math
import statistics as statistics
from scipy import stats

kernel = np.ones((3,3),np.uint8)

def find_keys(input_image):
	#INPUT: cropped image of piano keyboard
	#OUTPUT: contours of the black keys in the image, contours of the white keys in the image
	#cv2.imshow('temp',input_image)
	#cv2.waitKey(0)
	black_keys = black_key_search(input_image)
	white_keys = white_key_search(input_image,black_keys)
	return (black_keys, white_keys)

def black_key_search(image):
	print("={ Finding black keys }=")
	grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#Gaussian threshold:
	dst = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,71,0)
	#Morphological opening: erosion the dilation
	dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
	#========================================================
	#additional erosion:
	dst = cv2.erode(dst, kernel, iterations=1)
	#get the contours
	black_keys = find_black_key_contours(dst)
	if black_keys != -1 and len(black_keys) == 36:
		return sorted(black_keys, key = lambda cnt: cv2.boundingRect(cnt)[0])
	else:
		raise Exception("black key search failed to find 36 black keys")

def find_black_key_contours(thresh_im):
	"""
		INPUT: binary thresholded image
		OUTPUT: list of contours where each contour is the mask for a black key. Ordering = right most key to left most key

		METHOD: 1. find the contours using cv2.findContours
				2. filter them based on aspect ratio: they must be a certain amount longer in the y axis than the x axis to be key-shaped
				3. filter them based on size: remove any too small or too large to be valid keys
				4. filter them based on alignment: take the average top and bottom most points of the contours,
					any contours that have top of bottom points significantly different from this average can't be keys as the top and bottom of keys must form two straight lines
	"""
	#calculate size boundaries for black, keys based on image size:
	im_height, im_width = thresh_im.shape[:2]
	#max black key size = max white key size (area / 52 keys) * ratio of white to black key size * offset (for perspective correct)
	max_black_key_size = (im_height * im_width / 40) * 0.4 * 1.5

	#1. find the contours:
	contours, hierarchy = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#2, 3 & 4. now filter them:
	valid_contours = []
	for contour in contours:
		x,y,w,h = cv2.boundingRect(contour)
		aspect_ratio = float(w)/h
		area = cv2.contourArea(contour)
		#as black keys have a specific shape/aspect ratio
		if aspect_ratio >= 0.1 and aspect_ratio <= 0.3:
			if area >= ((1/8) * max_black_key_size) and area <= max_black_key_size:
				valid_contours.append(contour)
	if(len(valid_contours)>10):
		#EXTRA TO METHOD IN PAPER: now make sure all contours lie with topmost and bottommost points in a line:
		#this is only worth doing if there are a decent number of blobs found already anyway:
		#assume that most points are correct so find average topmost and bottommost point:
		avg_top_point = 0
		avg_bot_point = 0
		for d_cnt in valid_contours:
			top_point = tuple(d_cnt[d_cnt[:,:,1].argmin()][0])[1]
			bot_point = tuple(d_cnt[d_cnt[:,:,1].argmax()][0])[1]
			avg_top_point += top_point
			avg_bot_point += bot_point
		avg_top_point = avg_top_point / len(valid_contours)
		avg_bot_point = avg_bot_point / len(valid_contours)

		#now cycle through removing any blobs who's top or bottom points are too far off the mean values:
		trimmed_valid_contours = []
		plus_minus = 10
		for d_cnt in valid_contours:
			top_point = tuple(d_cnt[d_cnt[:,:,1].argmin()][0])[1]
			bot_point = tuple(d_cnt[d_cnt[:,:,1].argmax()][0])[1]
			if (top_point < avg_top_point + plus_minus and top_point > avg_top_point - plus_minus) and (bot_point < avg_bot_point + plus_minus and bot_point > avg_bot_point - plus_minus):
				trimmed_valid_contours.append(d_cnt)
		valid_contours = trimmed_valid_contours	
	return valid_contours

def white_key_search(image,black_keys):
	print("={ Finding white keys }=")
	#sharpening
	sharpening_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
	image = cv2.filter2D(image, -1, sharpening_kernel)
	#greyscale
	grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	#draw in all the black keys in pure black:
	cv2.drawContours(grey, black_keys, -1, (0,0,0), cv2.FILLED)
	#Gaussian thresh: medium/small $k$ with offset
	dst = cv2.adaptiveThreshold(grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)

	#houghlines
	lines = cv2.HoughLines(bottom_third(dst), 1, np.pi, 10)
	vertical_lines = []
	for line in lines:
		rho, theta = line[0]
		vertical_lines.append((rho,theta))

	#sort lines by x position (LHS to RHS):
	vertical_lines = sorted(vertical_lines, key = lambda line: np.cos(line[1]) * line[0])

	#get median line separation:
	prev_line_x = None
	distances = []
	for (rho,theta) in vertical_lines:
		x = int(np.cos(theta) * rho)
		if prev_line_x is None:
			prev_line_x = x
			distances.append(x)
		else:
			distances.append(x - prev_line_x)
			prev_line_x = x
	median_distance = statistics.median(distances)
	#improve the average calculation to a trimmed mean if enough outliers ignored:
	trimmed_mean = stats.trim_mean(distances, 0.1)
	if median_distance - 0.5 < trimmed_mean < median_distance + 0.5:
		median_distance = trimmed_mean
	
	#remove excess lines, add in any that are missing:
	prev_line_x = 0
	lines_to_keep = []
	for (rho,theta) in vertical_lines:
		dist_to_prev_line = rho - prev_line_x
		if median_distance * 0.8 < dist_to_prev_line < median_distance * 1.2:
			lines_to_keep.append((rho,theta))
			prev_line_x = rho
		elif dist_to_prev_line > median_distance * 1.8:
			lines_to_keep.append((rho,theta))
			prev_line_x = rho
			for xtra_line_val in range(0,int(round(dist_to_prev_line / median_distance) - 1)):
				new_rho = rho - int((xtra_line_val + 1) * median_distance)
				lines_to_keep.append((new_rho,0))

	#draw these key gaps onto the image:
	fully_segmented = draw_lines(grey,lines_to_keep,0)

	#threshold the image:
	ret, thresholded = cv2.threshold(fully_segmented,2,255,cv2.THRESH_BINARY)

	#find the contours:
	white_keys = find_white_key_contours(thresholded)

	if white_keys != -1 and len(white_keys) == 52:
		return sorted(white_keys, key = lambda cnt: cv2.boundingRect(cnt)[0])
	else:
		raise Exception("white key search failed to find 52 white keys")

def find_white_key_contours(thresh_im):
	#calculate size boundaries for black, keys based on image size:
	im_height, im_width = thresh_im.shape[:2]
	#max white key size = image area / 52 keys 
	max_white_key_size = (im_height * im_width / 52)

	#1. find the contours:
	contours, hierarchy = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#2, 3 & 4. now filter them:
	valid_contours = []
	for contour in contours:
		x,y,w,h = cv2.boundingRect(contour)
		aspect_ratio = float(w)/h
		area = cv2.contourArea(contour)
		#as white keys have a specific shape/aspect ratio
		if aspect_ratio >= 0.05 and aspect_ratio <= 0.3:
			if area >= ((1/5) * max_white_key_size):
				valid_contours.append(contour)
	return valid_contours

#===================
#Helper functions:
#===================

def bottom_third(frame):
	height, width = frame.shape[0], frame.shape[1]
	crop_frame = frame[int(height * 2/3):height, 0:width].copy()
	return crop_frame

def draw_lines(image,lines,colour):
	image = image.copy()
	for (rho,theta) in lines:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * rho
		y0 = b * rho
		x1 = int(x0 + 2000*(-b))
		y1 = int(y0 + 2000*(a))
		x2 = int(x0 - 2000*(-b))
		y2 = int(y0 - 2000*(a))
		cv2.line(image, (x1,y1),(x2,y2), colour, 1)
	return image