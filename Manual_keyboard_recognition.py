#Manual_keyboard_recognition.py
import numpy as np
import cv2 as cv2
import argparse
import os.path

class Keyboard_manual_find_and_transform:
	def __init__(self,initial_frame,target_dimensions,check=True):
		self.target_dimensions = target_dimensions
		self.p_mat = Manual_keyboard_recognition(initial_frame,target_dimensions,check)

	def transform_frame(self,frame):
		transformed_frame = cv2.warpPerspective(frame, self.p_mat, self.target_dimensions)
		return transformed_frame

#====================================================================================

def Manual_keyboard_recognition(frame,target_dimensions,extra_check):
	#attempt load from file:
	success, corners = load_from_file()
	#else do it manually:
	if not success:
		corners = Manual_corner_selection(frame)

	if extra_check:
		while True:
			p_mat = get_perspective_mat(corners,target_dimensions)
			transformed_frame = cv2.warpPerspective(frame, p_mat, target_dimensions)
			#allow user ability to manually reselect keyboard corners:
			cv2.imshow('Reselect keyboard region? (Y/N)',transformed_frame)
			key = cv2.waitKey(0) & 0xFF
			cv2.destroyWindow('Reselect keyboard region? (Y/N)')
			if key == ord("y"): 		
				corners = Manual_corner_selection(frame)
			else:
				print('={ Confirmed keyboard transformation }=')
				break
	p_mat = get_perspective_mat(corners,target_dimensions)
	return p_mat

def load_from_file():
	corners = []
	if os.path.exists('resources/saved_transformation.txt'):
		load_file = open("resources/saved_transformation.txt","r")
		contents = load_file.read()
		points_as_str = str.split(contents)
		for point in points_as_str:
			x = int(point.split(",")[0][1:])
			y = int(point.split(",")[1][:len(point.split(",")[1])-1])
			corners.append((x,y))
		load_file.close()
		return True, corners
	else:
		return False, None

def Manual_corner_selection(frame):
	corners = []
	while True:
		frame_copy = frame.copy()
		def point_select(self,event, x, y, flags, param):
			if event == cv2.EVENT_LBUTTONDOWN:
				corners.append((x,y))
				cv2.circle(frame_copy, (x,y), 2, (0,255,0), thickness = 1)
		cv2.imshow('Select keyboard region', frame_copy)
		cv2.setMouseCallback('Select keyboard region', point_select)
		key = cv2.waitKey(1) & 0xFF
		#if we press 'r' reset the image:
		if key == ord("r"):
			corners = []
			continue
		#if we press the t key, check that we have 4 points and then use that to transform the image:
		elif key == ord("t"):
			if len(corners) == 4:
				print('Saving and applying transform')
				break
			else:
				print('Incorrect number of points, image reset')
				corners = []
				continue
	#save these corners to the transformation file:
	transform_file = open('resources/saved_transformation.txt','w')
	for (x,y) in corners:
		transform_file.write("(" + str(x) + "," + str(y) + ") ")
	transform_file.close()
	return corners

def get_perspective_mat(corners,target_dimensions):
	(target_width, target_height) = target_dimensions
	target_frame = np.array([ [0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype = "float32")
	rect = np.zeros((4, 2), dtype = "float32")
	rect[0], rect[1], rect[2], rect[3] = corners[0], corners[1], corners[2], corners[3]
	p_mat = cv2.getPerspectiveTransform(rect,target_frame)
	#To transform using this matrix: warped_frame = cv2.warpPerspective(frame, p_mat, target_dimensions)
	return p_mat