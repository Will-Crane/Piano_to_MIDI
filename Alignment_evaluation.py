#Alignment_evaluation
import mido
import math

def Test_alignments(ground_truth_midi,predicted_midi):
	"""
		Inputs: ground truth midi collected and midi file extracted from system (both as file path strings)
		Output: Precision, Recall, alignment score

	"""
	truth_seq = MIDI_file_to_sequence(ground_truth_midi)
	test_seq = MIDI_file_to_sequence(predicted_midi)

	#test 61 different alignments (+-30 frames between the two):
	stats = []
	for shift in range(-6000,6001):
		if shift < 0:
			new_truth_seq = [set([])] * abs(shift * 3) + truth_seq
			new_test_seq = test_seq
		else:
			new_truth_seq = truth_seq
			new_test_seq = [set([])] * (shift * 2) + test_seq
		true_positives, false_positives, false_negatives = Alignment_evaluation(new_truth_seq,new_test_seq)
		stats.append((true_positives, false_positives, false_negatives))
	return stats


def Alignment_evaluation(ground_truth_midi,predicted_midi):
	"""
		Inputs: ground truth midi and midi file extracted from system (both as sequences; Lists of sets corresponding to MIDI events)
		Output: true_positives, false_positives, false_negatives

	"""
	true_positives = 0
	false_positives = 0
	false_negatives = 0

	#pad the shorter of the lists to make them the same length:
	ground_truth_midi += [set([])] * (max(len(ground_truth_midi),len(predicted_midi)) - len(ground_truth_midi))
	predicted_midi += [set([])] * (max(len(ground_truth_midi),len(predicted_midi)) - len(predicted_midi))

	for frame_num in range(0,len(ground_truth_midi)):
		#search for true positives and false negatives:
		for note_val in ground_truth_midi[frame_num]:
			if note_val in predicted_midi[frame_num]:
				true_positives += 1
			else:
				false_negatives += 1
		#search for false positives
		for note_val in predicted_midi[frame_num]:
			if note_val not in ground_truth_midi[frame_num]:
				false_positives += 1

	return true_positives, false_positives, false_negatives

def MIDI_file_to_sequence(midi_file):
	"""
		Input: MIDI file (as file path)
		Output: List of sets corresponding to MIDI events. 
			N.B. we are counting rests/periods of no notes being played towards this
			(with exception for the time before the first note)
			Format: outer list - each ms chunk, inner set - notes_on this ms chunk

	"""
	#1: convert the midi messages into using cumulative/absolute time (using first note time as 0s)
	#list of triples of (note_type, note_val, msg_time) where msg_time is cumulative/absolute
	cumulative_time = None
	for msg in mido.MidiFile(midi_file):
		if msg.type in ['note_on','note_off']:
			if cumulative_time is None:
				cumulative_time = []
				cumulative_time.append((msg.type,msg.note,0))
			else:
				cumulative_time.append((msg.type,msg.note,cumulative_time[-1][2] + msg.time))

	#print("Max time: " + str(cumulative_time[-1][2]))

	#2: convert to a dict from time chunk to list of msgs for that time chunk (rounding to convert time to nearest ms)
	frame_num_to_msgs = {}
	for (note_type,note_val,msg_time) in cumulative_time:
		rounded_time = math.ceil(msg_time * 1000)
		if rounded_time not in frame_num_to_msgs:
			frame_num_to_msgs[rounded_time] = []
		frame_num_to_msgs[rounded_time].append((note_type,note_val))

	#3: convert to list where index = time in ms and each item is a list of active notes
	toreturn = []
	max_frame_num = max(frame_num_to_msgs.keys())
	active_keys = set([])
	for frame_num in range(0,max_frame_num + 1):
		#first: process any messages:
		if frame_num in frame_num_to_msgs:
			for (msg_type, note_val) in frame_num_to_msgs[frame_num]:
				if msg_type == 'note_on' and note_val not in active_keys:
					active_keys.add(note_val)
					#print("ON: " + str(note_val))
				elif msg_type == 'note_off' and note_val in active_keys:

					active_keys.remove(note_val)
					#print("OFF: " + str(note_val))
		toreturn.append(active_keys.copy())
	if len(active_keys) != 0:
		raise Exception("invalid MIDI file: note left without note off" + active_keys)
	return toreturn

def Max_F1(pr_list):
	#Take in list of (true_positives, false_positives, false_negatives) elements
	#return (true_positives, false_positives, false_negatives, F1) that maximises F1 score
	max_vals = (0,0,0,0)
	for (true_positives, false_positives, false_negatives) in pr_list:
		precision = true_positives / (true_positives + false_positives)
		recall = true_positives / (true_positives + false_negatives)
		if (precision + recall) > 0:
			F1 = 2 * precision * recall / (precision + recall)
			if F1 > max_vals[3]:
				max_vals = (true_positives, false_positives, false_negatives, F1)
	return max_vals

def Find_best_F1(ground_truth_midi,predicted_midi):
	prec_list = Test_alignments(ground_truth_midi,predicted_midi)
	return Max_F1(prec_list)