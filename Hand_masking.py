#Hand_masking.py
import cv2 as cv2

#Thresholds:
H_thresh = 20
S_thresh = 83
R_thresh = 100
G_thresh = 50
B_thresh = 40

def find_hand_mask(frame):
	#Skin detection:
	RGB_mask = cv2.inRange(frame, (B_thresh,G_thresh,R_thresh), (255,255,255))

	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	h,s,v = cv2.split(hsv)
	ret, h_mask = cv2.threshold(h,H_thresh,255,cv2.THRESH_BINARY_INV)
	ret, s_mask = cv2.threshold(s,S_thresh,255,cv2.THRESH_BINARY)
	HSV_mask = cv2.bitwise_and(h_mask,s_mask)

	#OR masks together:
	combined_mask = cv2.bitwise_and(RGB_mask,HSV_mask)

	#Morphological transformation: opening followed by closing:
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
	combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
	combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
	
	return cv2.bitwise_not(combined_mask)