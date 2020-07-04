#MIDI_Handler
import time
import math
import numpy as np
from mido import Message, MetaMessage, MidiFile, MidiTrack, second2tick, bpm2tempo

class MIDI_Handler:

	def __init__(self):
		#dictionary from time in ms to set of active_notes
		self.event_map = dict()
		#current estimate of maximum timestamp value (also acts as timestamp val of previous frame)
		self.max_timestamp = 0

	def take_in_events(self, notes_on, timestamp):
		#update the event_map for all milliseconds this frame covers:
		for index in range(math.ceil(self.max_timestamp),math.ceil(timestamp)):
			self.event_map[index] = notes_on
		self.max_timestamp = timestamp

	def filter_minimum_note_length(self, min_note_length):
		#N.B min_note_length is in ms (not frames)
		#Assumption: we have read in  the entire file and know the max_timestamp
		#Assumption: We have already filtered out short gaps within a note
		for ms_index in range(0,math.ceil(self.max_timestamp)):
			#list of valid keys that will be included in self.event_map[timestamp]
			filtered_keys = []
			for key in self.event_map[ms_index]:
				#We check each note only when it is first pressed: if it is listed as on in the previous ms_index then the whole note is valid:
				if ms_index > 0 and key in self.event_map[ms_index - 1]:
					filtered_keys.append(key)
				else:
					#check that there are at least "min_note_length - 1" ms ahead where the key is listed as pressed:
					ms_with_key_unpressed = False
					for s_fn in range(ms_index + 1, min(ms_index + min_note_length,math.ceil(self.max_timestamp))):
						 if key not in self.event_map[s_fn]:
						 	ms_with_key_unpressed = True
					if not ms_with_key_unpressed:
						filtered_keys.append(key)
			self.event_map[ms_index] = filtered_keys

	def filter_note_gaps(self, max_note_gap):
		#max note gap = maximum gap (in ms) between two "ON" events for the same key for which we consider it to be part of the same single key press
		#Assumption: we have read in  the entire file and know the max_timestamp
		for ms_index in range(0,math.ceil(self.max_timestamp)):
			for key in self.event_map[ms_index]:
				#we check gaps only if this key press is the end of a note:
				if ms_index < math.ceil(self.max_timestamp) - max_note_gap and key not in self.event_map[ms_index + 1]:
					bridge_gap_to = -1
					for offset in range(1, max_note_gap + 1):
						if key in self.event_map[ms_index + offset]:
							bridge_gap = offset
							break
					if bridge_gap_to != -1:
						for bridge_offset in range(1, bridge_gap_to):
							self.event_map[ms_index + bridge_offset].append(key)
						
	def write_MIDI_file(self, video_file_path):
		#create the file path for the output midi file: (assume video in mp4 format and in resources/input_videos)
		output_file_path = 'resources/output_midi/' + video_file_path[23:-4] + '.mid'
		#initialize the MIDI file header:
		mid = MidiFile(ticks_per_beat=500)
		track = MidiTrack()
		mid.tracks.append(track)
		ticks_per_beat = 500
		tempo = bpm2tempo(120)
		track.append(MetaMessage('set_tempo', tempo=bpm2tempo(120)))
		
		last_tick_with_write = 0
		#special case for first frame:
		none_written = True
		for key in self.event_map[0]:
			midi_file.note_on(0, key, 64)
			none_written = False
		#iterate through the event_map:
		for ms_index in range(1,math.ceil(self.max_timestamp)):
			#Time-keeping:
			ticks = ms_index - last_tick_with_write

			#NOTE OFF events:
			for key in set(self.event_map[ms_index - 1]):
				if key not in self.event_map[ms_index]:
					track.append(Message('note_off', note=key, velocity=64, time=ticks))
					none_written = False
					ticks = 0

			#NOTE ON events:
			for key in set(self.event_map[ms_index]):
				if key not in self.event_map[ms_index - 1]:
					track.append(Message('note_on', note=key, velocity=64, time=ticks))
					none_written = False
					ticks = 0

			#Time-keeping:
			if not none_written:
				last_tick_with_write = ms_index
				none_written = True
		
		mid.save(output_file_path)
		print("={ MIDI file successfully written }=")