import cv2
import time
import os 
import sys
import yaml
import supervision as sv
import numpy as np

from src.detector import YOLOv5Detector
from src.tracker import DeepSortTracker
from src.dataloader import cap
from src.utils import get_iou, display_overlap


# Parameters from config.yml file
with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['main']


with open('config.yml' , 'r') as f:
    datal =yaml.safe_load(f)['yolov5_deepsort']['dataloader']


coordinates_file_path = datal['coordinates_path']


# poligon coordinates and zone type
polygons_cords = np.load(coordinates_file_path, allow_pickle=True) # loading coordinates and classes of zones from segmented_Areas file

zone_types = polygons_cords[-1] # the zone classes are saved
print(zone_types)

polygons_cords = np.delete(polygons_cords, -1)  # removing zone classes from the list of coordinates
polygons_cords = np.array(polygons_cords)

# Add the src directory to the module search path
sys.path.append(os.path.abspath('src'))

# Get YOLO Model Parameter
YOLO_MODEL_NAME = config['model_name']

# Visualization Parameterss

# Parameters from config.yml file
with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['main']


with open('config.yml' , 'r') as f:
    datal =yaml.safe_load(f)['yolov5_deepsort']['dataloader']



# Add the src directory to the module search path
sys.path.append(os.path.abspath('src'))

# Get YOLO Model Parameter
YOLO_MODEL_NAME = config['model_name']

# Visualization Parameters
DISP_FPS = config['disp_fps'] 
DISP_OBJ_COUNT = config['disp_obj_count']

object_detector = YOLOv5Detector(model_name=YOLO_MODEL_NAME)
#yolo_detector = YOLOv5Detector(model_name = "yolov5l")
#yolo_detector.set_bb_color((101, 122, 224))
tracker = DeepSortTracker()

track_history = {}    # Define a empty dictionary to store the previous center locations for each track ID

video_info = sv.VideoInfo.from_video_path(datal['data_path'])


output_path = "./output_videos"
video_name = output_path + "output.mp4"
fps = video_info.fps  # Frame per secondo
frame_size = (video_info.width, video_info.height)  # Dimensione dei frame
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per il video
out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

frame_number = 0

while cap.isOpened():

    detections = []

    success, img = cap.read() # Read the image frame from data source 
    frame_number += 1
 
    start_time = time.perf_counter()    #Start Timer - needed to calculate FPS
    
    # Object Detection
    results = object_detector.run_yolo(img)  # run the yolo v5 object detector 
    #results_yolo = yolo_detector.run_yolo(img)

    custom_detections , num_objects, class_ids = object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected
    #yolo_detections, yolo_num_objects, _ = yolo_detector.extract_detections(results_yolo, img, height=img.shape[0], width=img.shape[1])
    #print("before: ", class_ids, " + ", )
    # print(class_ids, " - ", yolo_num_objects)
    if False:
        detections = []
        updated_custom_detections = custom_detections[:]  
        for yolo_index, yolo_det in enumerate(yolo_detections):
            bb1 = yolo_det[0]
            to_add = True

            for custom_index, custom_det in enumerate(custom_detections):
                bb2 = custom_det[0]
                iou_result = get_iou(bb1, bb2)

                if iou_result > 0.4:
                    #display_overlap(bb1, bb2, iou_result, img)

                    if custom_det[2] == 'person-wheelchair':  # Classe predominante, nessuna sostituzione
                        to_add = False
                        break  # Non ha senso continuare a controllare altre detection

                    elif custom_det[1] < 0.80:  # Soglia di confidenza
                        updated_custom_detections[custom_index] = yolo_det
                        to_add = False
                        break

            if to_add:  
                detections.append(yolo_det)

        # Aggiungi le detection aggiornate del modello personalizzato
        detections.extend(updated_custom_detections)
        #detections = yolo_detections
        class_ids = []
        for det in detections:
            class_ids.append(det[2])
        
                        
        print(detections)
        #print(yolo_detections)
        '''
        for yolo_det, det in zip(yolo_detections, detections):
            yolo_label, label = yolo_det[2], det[2]
            print(f"{yolo_label} - {label}")
        '''

    # Object Tracking
    #print(detections)

    #print(class_ids)
    tracks_current = tracker.object_tracker.update_tracks(custom_detections, frame=img)
    tracker.display_track(track_history , tracks_current , img, class_ids, frame_number)
    
    # FPS Calculation
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = 1 / total_time

    #print(detections)

    # Descriptions on the output visualization
    cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'Frame: {int(frame_number)}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    

    #cv2.putText(img, f'MODEL: {YOLO_MODEL_NAME}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'TRACKED CLASSES: {object_detector.tracked_classes}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'TRACKER: {tracker.algo_name}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'DETECTED OBJECTS: {detections[2]}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    # draw the areas and show the number of violations in each of them
    for cords in polygons_cords:
        cords = np.array(cords)
        # draw areas
        color = (66, 215, 245) # RGB format
        cords_reshape = cords.reshape((-1, 1, 2))
        img = cv2.polylines(img, [cords_reshape], isClosed=True, color=color[::-1], thickness=2)

    cv2.imshow('img',img)
    out.write(img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# Release and destroy all windows before termination
cap.release()
out.release()

cv2.destroyAllWindows()


object_detector = YOLOv5Detector(model_name=YOLO_MODEL_NAME)
yolo_detector = YOLOv5Detector(model_name = "yolov5s", confidence = 0.9)
yolo_detector.set_bb_color((101, 122, 224))
tracker = DeepSortTracker()

track_history = {}    # Define a empty dictionary to store the previous center locations for each track ID

video_info = sv.VideoInfo.from_video_path(datal['data_path'])

output_path = "./output_videos"
video_name = output_path + "output.mp4"
fps = video_info.fps  # Frame per secondo
frame_size = (video_info.width, video_info.height)  # Dimensione dei frame
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per il video
out = cv2.VideoWriter(video_name, fourcc, 15, frame_size)


while cap.isOpened():
    detections = []

    success, img = cap.read() # Read the image frame from data source 
 
    start_time = time.perf_counter()    #Start Timer - needed to calculate FPS
    
    # Object Detection
    results = object_detector.run_yolo(img)  # run the yolo v5 object detector 
    results_yolo = yolo_detector.run_yolo(img)

    custom_detections , num_objects, class_ids = object_detector.extract_detections(results, img, height=img.shape[0], width=img.shape[1]) # Plot the bounding boxes and extract detections (needed for DeepSORT) and number of relavent objects detected
    yolo_detections, yolo_num_objects, _ = yolo_detector.extract_detections(results_yolo, img, height=img.shape[0], width=img.shape[1])

    #print(yolo_detections)
    '''
    for yolo_det, det in zip(yolo_detections, detections):
        yolo_label, label = yolo_det[2], det[2]
        print(f"{yolo_label} - {label}")
    '''

    # Object Tracking
    tracks_current = tracker.object_tracker.update_tracks(detections, frame=img)
    tracker.display_track(track_history , tracks_current , img, class_ids, frame_number)
    
    # FPS Calculation
    end_time = time.perf_counter()
    total_time = end_time - start_time
    fps = 1 / total_time

    #print(detections)


    # Descriptions on the output visualization
    #cv2.putText(img, f'FPS: {int(fps)}', (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'MODEL: {YOLO_MODEL_NAME}', (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'TRACKED CLASSES: {object_detector.tracked_classes}', (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'TRACKER: {tracker.algo_name}', (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'DETECTED OBJECTS: {num_objects}', (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #cv2.putText(img, f'DETECTED OBJECTS: {detections[2]}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    #cv2.imshow('img',img)
    #out.write(img)


    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break



# Release and destroy all windows before termination
cap.release()
out.release()

cv2.destroyAllWindows()

