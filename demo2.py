
from __future__ import division, print_function, absolute_import

import os
from timeit import time
import warnings
import sys
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
#from tqdm import tqdm
from cv2 import CAP_PROP_FRAME_COUNT
import imutils
import csv

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet

warnings.filterwarnings('ignore')

track_list = []
total=0
#counted = []
temp_up_list = []
temp_down_list = []
up_list = [0, 0, 0]
down_list = [0, 0, 0]


def Counting_People(yolo):
	global track_list
	global total
	#global total_frame
	currentFrame = 0
	#global currentFrame
	# Definition of the parameters
	max_cosine_distance = 0.3
	nn_budget = None
	nms_max_overlap = 1.0

	# deep_sort
	model_filename = 'model_data/vehicle.pb'
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)

	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	tracker = Tracker(metric)
	writeVideo_flag = True

	# here define your video file path 

	video_capture = cv2.VideoCapture('cam4.avi')

	#total_frame = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

	if writeVideo_flag:
		# Define the codec and create VideoWriter object
		w = int(video_capture.get(3))
		h = int(video_capture.get(4))
		fourcc = cv2.VideoWriter_fourcc(*'MJPG')
		out = cv2.VideoWriter('output1.avi', fourcc, 15, (w, h))
		list_file = open('detection.txt', 'w')
		frame_index = -1

	fps = 0.0
	currentFrame = 0
	frame_number=1


	while(video_capture.isOpened()):
		ret, frame = video_capture.read()  # frame shape 640*480*3
		

		up_line_position = 150
		middle_line_position = 180
		down_line_position = 210

		cv2.line(frame, (0,up_line_position), (w, up_line_position), (0, 0, 255), 1)
		cv2.line(frame, (0,middle_line_position), (w, middle_line_position), (0, 0, 255), 1)
		cv2.line(frame, (0,down_line_position), (w, down_line_position), (0, 0, 255), 1)

		if ret != True:
			break;
		t1 = time.time()
		currentFrame += 1
		image = Image.fromarray(frame)
		boxs = yolo.detect_image(image)
		# print("box_num",len(boxs))
		boxss = []
		labels = []
		for box in boxs:
			boxss.append(box[0:4])
			labels.append(box[4])
		features = encoder(frame, boxss)

		# score to 1.0 here).
		detections = [Detection(bbox, 1.0, feature, label) for bbox, feature, label in zip(boxss, features, labels)]

		# Run non-maxima suppression.
		boxes = np.array([d.tlwh for d in detections])
		scores = np.array([d.confidence for d in detections])
		indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
		detections = [detections[i] for i in indices]
		#print(detections)

		# Call the tracker
		tracker.predict()
		tracker.update(detections)

		for track in tracker.tracks:
			if track.is_confirmed() and track.time_since_update > 1:
				continue
				# count=0
			bbox = track.to_tlbr()
			a=int(bbox[0])
			b=int(bbox[1])
			c=int(bbox[2])
			d=int(bbox[3])

			if track.label == 1:
				index = 0
				class_name = 'Motor'
			elif track.label == 2:
				index = 1
				class_name = 'Car'
			elif track.label == 3:
				index = 2
				class_name = 'Truck'
			
			# Store Detections for MOTA chalenge Format
			# list1=[frame_number,track.track_id,a,b,c,d,-1,-1,-1,-1]

			# with open("res.txt", "a") as fp:
			# 	wr = csv.writer(fp, dialect='excel')
			# 	wr.writerow(list1)

			# (X, Y) = (int(box[0]), int(box[1]))
			# (W, H) = (int(box[2]), int(box[3]))
			
			xMid = int(int(bbox[0]) + (int(bbox[2])-int(bbox[0]))/2)
			yMid = int(int(bbox[1]) + (int(bbox[3])-int(bbox[1]))/2)

			

			if yMid > up_line_position and yMid < middle_line_position:
				if int(track.track_id) not in temp_up_list:
					temp_up_list.append(int(track.track_id))

			elif yMid < down_line_position and yMid > middle_line_position:
				if int(track.track_id) not in temp_down_list:
					temp_down_list.append(int(track.track_id))

			if yMid > middle_line_position:
				if int(track.track_id) in temp_up_list:
					temp_up_list.remove(track.track_id)
					down_list[index] += 1

			elif yMid < middle_line_position:
				if int(track.track_id) in temp_down_list:
					temp_down_list.remove(track.track_id)
					up_list[index] += 1



			# if yMid > 450 and yMid < 550:
			# 	if int(track.track_id) not in counted:
			# 		print(int(track.track_id))
			# 		print(counted)
			# 		vehicle += 1
			# 		counted.append(int(track.track_id)) 

			

			cv2.circle(frame, (xMid, yMid), 5, (0,0,255), 5)
			cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
			cv2.putText(frame, str(track.track_id)+ ' ' + str(class_name), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 2)
			#cv2.putText(frame, 'Total Vehicles: {}.'.format(vehicle), (450, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

			cv2.putText(frame, "Up", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
			cv2.putText(frame, "Down", (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
			cv2.putText(frame, "Car:        "+str(up_list[1])+"     "+ str(down_list[1]), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
			cv2.putText(frame, "Motor:      "+str(up_list[0])+"     "+ str(down_list[0]), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
			cv2.putText(frame, "Truck:      "+str(up_list[2])+"     "+ str(down_list[2]), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

			track_list.append((track.track_id, track.label))

		frame_number+=1
		for det in detections:
			bbox = det.to_tlbr()
			
			
		cv2.imshow('', frame)

		if writeVideo_flag:
			# save a frame
			out.write(frame)
			frame_index = frame_index + 1

			list_file.write(str(frame_index) + ' ')

			if len(boxs) != 0:
				for i in range(0, len(boxs)):
					list_file.write(str(boxs[i][0]) + ' ' + str(boxs[i][1]) + ' ' + str(boxs[i][2]) + ' ' + str(
						boxs[i][3]) + ' ')
			list_file.write('\n')

	
		fps = (fps + (1. / (time.time() - t1))) / 2
		# print (my_list)
		total = (len(set(track_list)))
		print("fps= %f" % (fps))
		# Press Q to stop!
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	video_capture.release()
	if writeVideo_flag:
		out.release()
		list_file.close()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	Counting_People(YOLO())
