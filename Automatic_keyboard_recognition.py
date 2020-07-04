#Automatic_keyboard_recognition
import numpy as np
import math
import statistics as stats
import cv2 as cv2

class Keyboard_auto_find_and_transform:
	def __init__(self,initial_frame,target_dimensions):
		print("Finding Keyboard")
		self.target_dimensions = target_dimensions
		self.p_mat = Automatic_keyboard_recognition(initial_frame,target_dimensions)
		print("Keyboard found")

	def transform_frame(self,frame):
		transformed_frame = cv2.warpPerspective(frame, self.p_mat, self.target_dimensions)
		return transformed_frame

#====================================================================================

def Automatic_keyboard_recognition(frame,target_dimensions):
	lines_within_frame = get_lines(frame.copy())
	horizontal_ish_lines = filter_horizontal_lines(lines_within_frame)
	#list of line pairings: each pairing in form ((r1,theta1),(r2,theta2)) where (r1,theta1) is possible top of keyboard line
	line_pairings = group_lines(horizontal_ish_lines)
	filtered_pairings = filter_pairings(line_pairings,frame)
	valid_pairings = test_pairings(filtered_pairings,frame)
	#take smallest valid keyboard region:
	bounding_lines = None
	if len(valid_pairings) == 0:
		raise Exception('No Valid Keyboard found within region')
	else:
		bounding_lines = valid_pairings[0]
	corners = get_keyboard_corners(bounding_lines,frame.copy())
	save_corners_to_file(corners)
	perspective_matrix = get_perspective_mat(corners,target_dimensions)
	return perspective_matrix

def get_lines(frame):
	grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#Canny parameters: (image, weak_thresh, strong_thresh)
	edges = cv2.Canny(grey,70,150)
	#Hough parameters: (edge_im, rho_resolution, theta_resolution, hough space threshold)
	lines = cv2.HoughLines(edges, 1, np.pi/180, 180)
	#convert to a list of (rho,theta) elements (no additional information required):
	formatted_lines = []
	for line in lines:
		rho, theta = line[0]
		formatted_lines.append((rho,theta))

	return formatted_lines

def filter_horizontal_lines(lines):
	filtered_lines = []
	for (rho,theta) in lines:
		if theta > np.pi/3 and theta < np.pi*2/3:
			filtered_lines.append((rho,theta))
	return filtered_lines

def group_lines(lines):
	line_pairings = []
	grouped_lines = []
	num_lines = len(lines)
	while len(grouped_lines) < num_lines:
		ungrouped_lines = [x for x in lines if x not in grouped_lines]
		base_line = ungrouped_lines[0]
		rho_b, theta_b = base_line
		grouped_lines.append(base_line)
		for rho_c, theta_c in ungrouped_lines[1:]:
			if abs(theta_c - theta_b) < np.pi/30:
				line_pairings.append(((rho_b,theta_b),(rho_c,theta_c)))
	return line_pairings

def filter_pairings(pairings,frame):
	#remove pairings that are too close together at x=xmax/2 AND order within pairings such that higher line at xmax/2 is first):
	filtered_pairings = []
	height = frame.shape[0]
	xmid = int(frame.shape[1] / 2)
	# y = (r - xcos(theta)/sin(theta))
	for (line1,line2) in pairings:
		y1 = (line1[0] - xmid * np.cos(line1[1])) / np.sin(line1[1])
		y2 = (line2[0] - xmid * np.cos(line2[1])) / np.sin(line2[1])
		if abs(y1 - y2) > height / 70:
			#order the pairing from heighest to lowest (N.B. OpenCV origin is at top left)
			if y1 > y2:
				filtered_pairings.append((line2,line1))
			else:
				filtered_pairings.append((line1,line2))
	return filtered_pairings

def test_pairings(pairings,frame):
	valid_pairings = []
	height,width = frame.shape[:2]
	for (line1,line2) in pairings:
		#find intercepts at x=0 and x=xmax
		yA = line1[0]/np.sin(line1[1])
		yB = (line1[0] - width * np.cos(line1[1]))/np.sin(line1[1])
		yC = line2[0]/np.sin(line2[1])
		yD = (line2[0] - width * np.cos(line2[1]))/np.sin(line2[1])
		#crop in to the lines and warp to make them parallel:
		target_frame = np.array([ [0, 0], [999, 0], [999, 98], [0, 98]], dtype = "float32")
		rect = np.zeros((4, 2), dtype = "float32")
		rect[0], rect[1], rect[2], rect[3] = (0,yA), (width,yB), (width,yD), (0,yC)
		p_mat = cv2.getPerspectiveTransform(rect,target_frame)
		warped_frame = cv2.warpPerspective(frame.copy(), p_mat, (1000,99))
		if brightness_test(warped_frame):
			test_result, black_keys = black_key_test(warped_frame)
			if test_result:		
				#draw the black keys in pure black on the warped_frame:
				cv2.drawContours(warped_frame, black_keys, -1, (255,255,255), cv2.FILLED)
				inverse_p_mat = cv2.getPerspectiveTransform(target_frame,rect)
				valid_pairings.append((line1,line2,inverse_p_mat,warped_frame))
				break
	return valid_pairings

def brightness_test(transformed_frame):
	#return bool, in this cropped and transformed_frame, is the bottom third lighter than both the middle and top thirds?
	height, width = transformed_frame.shape[:2]
	q1 = int(height/3)
	q2 = int(height * 2/3)

	top_third = transformed_frame[0:q1,0:width]
	avg_color_per_row = np.average(top_third, axis=0)
	avg_color_top_third = np.average(avg_color_per_row, axis=0)

	middle_third = transformed_frame[q1:q2,0:width]
	avg_color_per_row = np.average(middle_third, axis=0)
	avg_color_middle_third = np.average(avg_color_per_row, axis=0)

	bottom_third = transformed_frame[q2:height,0:width]
	avg_color_per_row = np.average(bottom_third, axis=0)
	avg_color_bottom_third = np.average(avg_color_per_row, axis=0)

	#RGB -> Luma conversion:https://stackoverflow.com/questions/596216/formula-to-determine-brightness-of-rgb-color
	Ltop = 0.375 * avg_color_top_third[0] + 0.5 * avg_color_top_third[1] + 0.16 * avg_color_top_third[2]
	Lmid = 0.375 * avg_color_middle_third[0] + 0.5 * avg_color_middle_third[1] + 0.16 * avg_color_middle_third[2]
	Lbot = 0.375 * avg_color_bottom_third[0] + 0.5 * avg_color_bottom_third[1] + 0.16 * avg_color_bottom_third[2]

	if Lbot > Lmid and Lbot > Ltop:
		return True
	else:
		return False

def black_key_test(transformed_frame):
	grey = cv2.cvtColor(transformed_frame,cv2.COLOR_BGR2GRAY)
	#exhaustive search of thresholds
	lowThres, highThres = 0,254
	maxBlackno, optimalThres = 0, 0
	black_keys = None
	while highThres > lowThres:
		ret, dst = cv2.threshold(grey,lowThres,255,cv2.THRESH_BINARY_INV)
		black_keys = custom_black_key_contours(dst)
		if len(black_keys) >= maxBlackno and len(black_keys) <= 36:
			maxBlackno = len(black_keys)
			optimalThres = lowThres
		lowThres += 1
	ret, dst = cv2.threshold(grey,optimalThres,255,cv2.THRESH_BINARY_INV)
	black_keys = custom_black_key_contours(dst)

	if len(black_keys) == 36:
		return True, black_keys
	else:
		return False, black_keys

def custom_black_key_contours(thresh_im):
	#find the black key contours:
	contours, hierarchy = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#get median size of contours:
	contour_sizes = []
	for contour in contours:
		area = cv2.contourArea(contour)
		contour_sizes.append(area)
	median_size = stats.median(contour_sizes)

	#filter so that they are all roughly the same size and with valid aspect ratio and their bottom points aren't too low:
	height, width = thresh_im.shape[:2]
	filtered_contours = []
	for contour in contours:
		area = cv2.contourArea(contour)
		x,y,w,h = cv2.boundingRect(contour)
		aspect_ratio = float(w)/h
		if (median_size / 2 < area < median_size * 2) and aspect_ratio < 0.3 and (y + h) < height * 3/4:
			filtered_contours.append(contour)
	return filtered_contours

def get_keyboard_corners(pairing,frame):
	(line1,line2,inverse_p_mat,warped_frame) = pairing
	height,width = frame.shape[:2]

	#undo warping to find black_key contours with respect to original frame:
	unwarped = cv2.warpPerspective(warped_frame.copy(), inverse_p_mat, (width,height))
	ret, dst = cv2.threshold(cv2.cvtColor(unwarped,cv2.COLOR_BGR2GRAY),254,255,cv2.THRESH_BINARY)
	black_keys, hierarchy = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#filter out any small artefacts/noise:
	filtered_black_keys = []
	total_bk_area = 0
	for black_key in black_keys:
		key_area = cv2.contourArea(black_key)
		if key_area > 20:
			total_bk_area += key_area
			filtered_black_keys.append(black_key)
	black_keys = filtered_black_keys
	if len(black_keys) != 36:
		raise Exception("Black Key lost for corner detection")

	#order black keys from LHS to RHS:
	black_keys = sorted(black_keys, key = lambda cnt: cv2.boundingRect(cnt)[0])
	
	#draw a white line to connect the lowest and highest white keys:
	left_most_point = np.array(list(black_keys[0][black_keys[0][:,:,0].argmin()][0]))
	Bb6_centre = np.array(list(get_contour_centre(black_keys[-1])))
	top_bk_separation = distance_between_points(get_contour_centre(black_keys[-2]),get_contour_centre(black_keys[-1]))
	Top_line_point = Bb6_centre + top_bk_separation * (4/3) * (Bb6_centre - left_most_point)/(np.linalg.norm(Bb6_centre - left_most_point))
	tmp = (int(Top_line_point[0]),int(Top_line_point[1]))
	lmp = (left_most_point[0] - 1,left_most_point[1] + 1)
	cv2.line(frame,lmp,tmp,(255,255,255),2)

	#threshold the image to find a contour containing all the white keys: 
	#centre must be in valid range of middle of the keyboard: 
	#area must be greater than total bk area:

	#get valid centre range:
	x_range = (get_contour_centre(black_keys[15])[0], get_contour_centre(black_keys[25])[0])
	y_max = max(tuple(black_keys[0][black_keys[0][:,:,1].argmax()][0])[1], tuple(black_keys[-1][black_keys[-1][:,:,1].argmax()][0])[1])
	y_min = min(tuple(black_keys[0][black_keys[0][:,:,1].argmin()][0])[1], tuple(black_keys[-1][black_keys[-1][:,:,1].argmin()][0])[1])
	y_range = (y_min, y_max)


	grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	highThresh = 254
	optimalThres = 0
	while highThresh > 0:
		ret, dst = cv2.threshold(grey,highThresh,255,cv2.THRESH_BINARY_INV)
		white_key_contour = custom_white_key_contour(dst,x_range,y_range)
		if white_key_contour is not None and total_bk_area * 3 < cv2.contourArea(white_key_contour) < total_bk_area * 3.15:
			optimalThres = highThresh
		highThresh -= 1
	ret, dst = cv2.threshold(grey,optimalThres,255,cv2.THRESH_BINARY_INV)
	white_key_contour = custom_white_key_contour(dst,x_range,y_range)
	#draw this contour that covers all white keys in pure white on the frame:
	cv2.drawContours(frame,[white_key_contour],-1,(255,255,255),cv2.FILLED)
	#bounding rect for this contour:
	x,y,w,h = cv2.boundingRect(white_key_contour)
	corners = []
	for origin_point in [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]:
		corner = closest_white_pixel(frame,origin_point)
		corners.append((corner[0],corner[1]))
	return corners

def custom_white_key_contour(thresh_im,x_range,y_range):
	#xrange, yrange are range of contour centre that would be valid
	(xmin, xmax) = x_range
	(ymin, ymax) = y_range
	contours, hierarchy = cv2.findContours(thresh_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if len(contours) == 0:
		return None
	else:
		#take largest contour (in range) only:
		largest = None
		size_of_largest = 0
		for contour in contours:
			(Cx,Cy) = get_contour_centre(contour)
			if (xmin < Cx < xmax) and (ymin < Cy < ymax) and cv2.contourArea(contour) > size_of_largest:
				largest = contour
				size_of_largest = cv2.contourArea(contour)
		return largest

def closest_white_pixel(image,origin_point):
	"""
		Inputs: image (here taking in image with all keys masked in pure white and line connecting all keys), point in image from which to search
		output: (x,y) of closest pixel of colour (255,255,255) to the point in the image

		METHOD: starting with radius 1, search from the bounding box point for a pure white pixel,
				if found, return this pixel position,
				else, increment the search circle radius and try again
				(N.B. if multiple white pixels found, take the one closest to the bounding box corner)
	"""
	(px,py) = origin_point
	im_h, im_w, im_d = image.shape
	#keyboard mask: keyboard in white, all else black
	mask1 = cv2.inRange(image, (255,255,255), (255,255,255))
	for radius in range(0,im_h):
		#create a mask for the radius in which we are searching
		circle_mask = np.zeros((im_h,im_w),dtype=image.dtype)
		cv2.circle(circle_mask,(px,py),radius,(255,255,255),thickness=-1)
		#combine this with the keyboard mask: combined_mask = all black except for any bit of keyboard that is within our search circle
		combined_mask = cv2.bitwise_and(mask1,circle_mask)
		formatted_mask = cv2.bitwise_and(image,image,mask=combined_mask)

		grey = cv2.cvtColor(formatted_mask,cv2.COLOR_BGR2GRAY)
		ret, dst = cv2.threshold(grey,254,255,cv2.THRESH_BINARY)
		white_pixels = cv2.findNonZero(dst)

		if white_pixels is not None:
			closest_pixel = None
			closest_pixel_distance = 10000
			for white_point in white_pixels:
				distance = distance_between_points(origin_point,white_point[0])
				if distance < closest_pixel_distance:
					closest_pixel_distance = distance
					closest_pixel = white_point[0]
			return closest_pixel	
	return (-1,-1)

def get_perspective_mat(corners,target_dimensions):
	(target_width, target_height) = target_dimensions
	target_frame = np.array([ [0, 0], [target_width - 1, 0], [target_width - 1, target_height - 1], [0, target_height - 1]], dtype = "float32")
	rect = np.zeros((4, 2), dtype = "float32")
	rect[0], rect[1], rect[2], rect[3] = corners[0], corners[1], corners[2], corners[3]
	p_mat = cv2.getPerspectiveTransform(rect,target_frame)
	#To transform using this matrix: warped_frame = cv2.warpPerspective(frame, p_mat, target_dimensions)
	return p_mat

def save_corners_to_file(corners):
	#save these corners to the transformation file:
	transform_file = open('resources/saved_transformation.txt','w')
	for (x,y) in corners:
		transform_file.write("(" + str(x) + "," + str(y) + ") ")
	transform_file.close()

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
		cv2.line(image, (x1,y1),(x2,y2), colour, 2)
	return image

def get_contour_centre(cnt):
	x,y,w,h = cv2.boundingRect(cnt)
	Cx = int(x + w/2)
	Cy = int(y + h/2)
	return (Cx,Cy)

def distance_between_points(point1,point2):
	(x1,y1) = point1
	(x2,y2) = point2
	dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
	return dist