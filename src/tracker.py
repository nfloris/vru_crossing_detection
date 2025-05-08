from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np
import yaml

from src.utils import get_polygon_center, get_rectangle_center, is_point_inside_polygon 
from collections import deque


with open('config.yml' , 'r') as f:
    config =yaml.safe_load(f)['yolov5_deepsort']['tracker']

#Visualization parameters

DISP_TRACKS = config['disp_tracks']
DISP_OBJ_TRACK_BOX = config['disp_obj_track_box']
OBJ_TRACK_COLOR = tuple(config['obj_tack_color'])
OBJ_TRACK_BOX_COLOR = tuple(config['obj_track_box_color'])

# # Deep Sort Parameters (check config.yml for parameter descriptions)
# MAX_AGE = config['max_age']   
# N_INIT =config['n_init']    
# NMS_MAX_OVERLAP = config['nms_max_overlap']       
# MAX_COSINE_DISTANCE = config['max_cosine_distance']    
# NN_BUDGET = config['nn_budget']            
# OVERRIDE_TRACK_CLASS = config['override_track_class'] 
# EMBEDDER = config['embedder']
# HALF = config['half'] 
# BGR = config['bgr']
# EMBEDDER_GPU = config['embedder_gpu'] 
# EMBEDDER_MODEL_NAME = config['embedder_model_name']    
# EMBEDDER_WTS = config['embedder_wts']           
# POLYGON = config['polygon']              
# TODAY = config['today']                  


# class DeepSortTracker(): 

#     def __init__(self):
        
#         self.algo_name ="DeepSORT"

#         self.object_tracker = DeepSort(max_age=config['max_age'] ,
#                 n_init=config['n_init'],
#                 nms_max_overlap=config['nms_max_overlap'],
#                 max_cosine_distance=config['max_cosine_distance'],
#                 nn_budget=config['nn_budget'],
#                 override_track_class=config['override_track_class'] ,
#                 embedder=config['embedder'],
#                 half=config['half'],
#                 bgr=config['bgr'],
#                 embedder_gpu=config['embedder_gpu'],
#                 embedder_model_name=config['embedder_model_name'] ,
#                 embedder_wts=config['embedder_wts'],
#                 polygon=config['polygon'],
#                 today=config['today'])

# Deep Sort Parameters
MAX_AGE = 15                 # Maximum number of frames to keep a track alive without new detections. Default is 30

N_INIT = 3                  # Minimum number of detections needed to start a new track. Default is 3

NMS_MAX_OVERLAP = 1.0       # Maximum overlap between bounding boxes allowed for non maximal supression(NMS).
                            #If two bounding boxes overlap by more than this value, the one with the lower confidence score is suppressed. Defaults to 1.0.

MAX_COSINE_DISTANCE = 0.1   # Maximum cosine distance allowed for matching detections to existing tracks. 
                            #If the cosine distance between the detection's feature vector and the track's feature vector is higher than this value, 
                            # the detection is not matched to the track. Defaults to 0.2

NN_BUDGET = None            # Maximum number of features to store in the Nearest Neighbor index. If set to None, the index will have an unlimited budget. 
                            #This parameter affects the memory usage of the tracker. Defaults to None.

OVERRIDE_TRACK_CLASS = None  #Optional override for the Track class used by the tracker. This can be used to subclass the Track class and add custom functionality. Defaults to None.
EMBEDDER = "mobilenet"       #The name of the feature extraction model to use. The options are "mobilenet" or "efficientnet". Defaults to "mobilenet".
HALF = True                  # Whether to use half-precision floating point format for feature extraction. This can reduce memory usage but may result in lower accuracy. Defaults to True
BGR = False                   #Whether to use BGR color format for images. If set to False, RGB format will be used. Defaults to True.
EMBEDDER_GPU = True          #Whether to use GPU for feature extraction. If set to False, CPU will be used. Defaults to True.
EMBEDDER_MODEL_NAME = None   #Optional model name for the feature extraction model. If not provided, the default model for the selected embedder will be used.
EMBEDDER_WTS = None          # Optional path to the weights file for the feature extraction model. If not provided, the default weights for the selected embedder will be used.
POLYGON = False              # Whether to use polygon instead of bounding boxes for tracking. Defaults to False.
TODAY = None                 # Optional argument to set the current date. This is used to calculate the age of each track in days. If not provided, the current date is used.


# retrieving coordinates filepath




class DetectorManager():

    def __init__(self):

        self.polygon_coords = None
        self.zone_types = None

        self.detections_in_zone = dict()
        self.pedestrians_at_risk = dict()

        self.num_samplings = 120
        self.frame_threshold = 50
        self.warning_duration = 100 # in frames

    
    def retrieve_polygons(self):

        with open('config.yml' , 'r') as f:
            datal = yaml.safe_load(f)['yolov5_deepsort']['dataloader']

        coordinates_file_path = datal['coordinates_path']

        # poligon coordinates and zone type
        self.polygons_cords = np.load(coordinates_file_path, allow_pickle=True) # loading coordinates and classes of zones from segmented_Areas file

        self.zone_types = self.polygons_cords[-1] # the zone classes are saved
        print(self.zone_types)

        self.polygons_cords = np.delete(self.polygons_cords, -1)  # removing zone classes from the list of coordinates
        self.polygons_cords = np.array(self.polygons_cords)

    
    def check_hazard(self, entity_id, entity_bbox, frame_number):
        
        hazard = False

        if entity_id not in self.detections_in_zone:
            self.detections_in_zone[entity_id] = deque(maxlen=self.num_samplings + 1)

        for i, polygon in enumerate(self.polygons_cords):
            if is_point_inside_polygon(entity_bbox, polygon):
                #print("entity ", entity_id , "INSIDE of polygon ", i)
                self.detections_in_zone[entity_id].append(1)
            else: 
                self.detections_in_zone[entity_id].append(0)



        
        risk_exposure_level = sum(self.detections_in_zone[entity_id])
        print(risk_exposure_level)
        if risk_exposure_level > self.frame_threshold:
            self.pedestrians_at_risk[entity_id] = frame_number
        
        if entity_id in self.pedestrians_at_risk:
            if frame_number - self.pedestrians_at_risk[entity_id] < self.warning_duration:
                hazard = True
            else: 
                hazard = False

        return hazard
    


    def generate_warning(self, img, frame_number):
        
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 30), (790, 120), (0, 0, 0), -1) 
        alpha = 0.7 
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        
        label = '   UTENTE DEBOLE IN PROCINTO DI ATTRAVERSARE   '

        if (frame_number // 15) % 2 == 0:
            cv2.putText(img, ' !', (20, 73), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(img, ' !', (740, 73), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(img, label, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)





class DeepSortTracker(): 

    def __init__(self):
        
        self.algo_name ="DeepSORT"
        self.object_tracker = DeepSort(max_age=MAX_AGE,
                n_init=N_INIT,
                nms_max_overlap=NMS_MAX_OVERLAP,
                max_cosine_distance=MAX_COSINE_DISTANCE,
                nn_budget=NN_BUDGET,
                override_track_class=OVERRIDE_TRACK_CLASS,
                embedder=EMBEDDER,
                half=HALF,
                bgr=BGR,
                embedder_gpu=EMBEDDER_GPU,
                embedder_model_name=EMBEDDER_MODEL_NAME,
                embedder_wts=EMBEDDER_WTS,
                polygon=POLYGON,
                today=TODAY)

        self.detectorManager = DetectorManager()
        self.detectorManager.retrieve_polygons()


    
        
    def display_track(self , track_history , tracks_current , img, class_ids, frame_number):
        

        for track, class_id in zip(tracks_current, class_ids):
            if not track.is_confirmed():
                continue
            track_id = track.track_id

            color = (18, 122, 43)
            
            # Retrieve the current track location(i.e - center of the bounding box) and bounding box
            location = track.to_tlbr()
            bbox = location[:4].astype(int)
            bbox_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            

            # checking on disabled pedestrians about to cross
            if self.detectorManager.check_hazard(track_id, bbox_center, frame_number):
                # if a disabled pedestrian is aproaching the road, then generate a real-time warning
                self.detectorManager.generate_warning(img, frame_number)
                # the pedestrian track is colored in red
                color = (31, 13, 191)
            
            # Retrieve the previous center location, if available
            prev_centers = track_history.get(track_id ,[])
            prev_centers.append(bbox_center)
            track_history[track_id] = prev_centers
            
            # Draw the track line, if there is a previous center location
            if prev_centers is not None and DISP_TRACKS == True:
                points = np.array(prev_centers, np.int32)
                cv2.polylines(img, [points], False, (51 ,225, 255), 2)

            if DISP_OBJ_TRACK_BOX == True: 
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                cv2.rectangle(img,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])), color ,1)


                label = f"{class_id} "# - {track_id}"

                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y_label = max(int(y1), label_size[1] + 10)
                cv2.rectangle(img, (int(x1), y_label - label_size[1] - 10), (int(x1) + label_size[0], y_label + base_line - 10), color, cv2.FILLED)
                cv2.putText(img, label, (int(x1), y_label - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            