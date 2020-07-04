For a full description of the architecture and analysis of system limitations, please consult the dissertation pdf.

N.B. 	This system is not pose invariant; there are strict requirements for test data to enable keyboard recognition.
	If these requirements are not met, please use manual keyboard recognition.

Conversion of video file to MIDI file:

Step 1: place the video file in the folder resources\input_videos
Step 2: call the Video_to_MIDI function from the Video_to_MIDI file.
	N.B. for video file path you must include "resources\input_videos"
	N.B. the parameter format required as the second argument need be a list: [threshold, min_height, min_area, top_point]
	N.B. the top_point parameter is suited for OpenCV where the origin is in the top left of the image
	N.B. the default system for MIDI file creation is MOG2, if median is requested, add mog2=False as an argument



Comparison of MIDI file with ground truth:

Step 1: ensure that the ground truth MIDI file has the same name as it's corresponding video file
Step 2: place the truth MIDI file in the resources/midi_truth folder
Step 3: Ensure that the MIDI file to compare is of the same name and present in the resources\output_midi folder
Step 4: Call the function "Find_best_F1" from the Alignment_evaluation file. 
	This requires the file paths of the MIDI files you want compared and outputs(true_positives, false_positives, false_negatives, F1) for those files with alignment such that F1 is maximised
