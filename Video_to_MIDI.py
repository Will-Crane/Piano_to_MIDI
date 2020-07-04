#Video_to_MIDI.py
import cv2 as cv2
import math
from MIDI_Handler import MIDI_Handler
from Automatic_keyboard_recognition import Keyboard_auto_find_and_transform
from Manual_keyboard_recognition import Keyboard_manual_find_and_transform
from Key_search import find_keys
from Hand_masking import find_hand_mask
from Key_press_detection_mog2 import Key_press_detection_mog2
from Key_press_detection_median import Key_press_detection_median

automatic_keyboard_detection = True
keyboard_detection_check = False
target_dimensions = (1000,150)

def Video_to_MIDI(video_file_path,params,mog2=True):
	#Inputs: video_file_path: file path to video file
	#params: list of format: [threshold, min_height, min_area, top_point]

	cap = cv2.VideoCapture(video_file_path)
	rough_fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
	if cap.isOpened()== False:
		raise Exception("Error opening video file")

	#=======================================================
	#INITIALISATION
	#=======================================================

	#take in first frame for initialisation:
	ret, initial_frame = cap.read()

	#MIDI object will store in the data and then write it to a midi file:
	midi = MIDI_Handler()
	#KB_transform finds keyboard from initialisation frame and then transforms all future frames:
	KB_transform = None
	if automatic_keyboard_detection:
		KB_transform = Keyboard_auto_find_and_transform(initial_frame.copy(),target_dimensions)
	else:
		KB_transform = Keyboard_manual_find_and_transform(initial_frame.copy(),target_dimensions,check=keyboard_detection_check)

	transformed_initial_frame = KB_transform.transform_frame(initial_frame.copy())

	(black_keys, white_keys) = find_keys(transformed_initial_frame.copy())

	KP_detect = None
	if mog2:
		KP_detect = Key_press_detection_mog2(transformed_initial_frame, (black_keys, white_keys), params)
	else:
		KP_detect = Key_press_detection_median(transformed_initial_frame, (black_keys, white_keys), params, history=900, sample_rate=10)

	#=======================================================
	#READ IN VIDEO FILE:
	#=======================================================

	print("={ Reading in video file }=")
	while cap.isOpened():
		ret, frame = cap.read()
		timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

		if ret == False:
			break

		transformed_frame = KB_transform.transform_frame(frame)
		hand_mask = find_hand_mask(transformed_frame)
		
		active_keys = KP_detect.detect_pressed_keys(transformed_frame,hand_mask)
		midi.take_in_events(active_keys, timestamp)

	#=======================================================
	#WRITE OUT MIDI FILE:
	#=======================================================

	#clean up windows and capture:
	cap.release()
	cv2.destroyAllWindows()

	#filter the data: (must convert from frames to ms (roughly))
	midi.filter_minimum_note_length(1 * rough_fps)
	midi.filter_note_gaps(1 * rough_fps)
	#midi.filter_minimum_note_length(2 * rough_fps)

	#write the MIDI file out:
	midi.write_MIDI_file(video_file_path)
