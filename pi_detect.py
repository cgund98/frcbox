import numpy as np
import os
import tensorflow as tf
import cv2
from picamera.array import PiRGBArray
from picamera.array import PiRGBAnalysis
from picamera import PiCamera
import sys
import time

# Import Gambezi (Our smart dashboard equivalent)
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from gambezi_python import Gambezi

width = 240
height = 320

# Import utilities
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

# Define import paths
PATH_TO_CKPT = 'output_inference_graph-1.4.1.pb/frozen_inference_graph.pb' # Import frozen model
PATH_TO_LABELS = 'frc_label_map.pbtxt' # Import map of labels
NUM_CLASSES=1 # Only one class (box)

# Load frozen model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Load Label Map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Helper function for data format
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Define function for detecting objects within an image
def detect_objects(image_np, sess, detection_graph):
  #Define input
  image_np_expanded = np.expand_dims(image_np, axis=0)
  image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
  
  #Define outputs
  detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
  detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
  detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  
  options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  run_metadata = tf.RunMetadata()
  
  #Predict
  (boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_np_expanded})
  
  #Visualize
  #vis_util.visualize_boxes_and_labels_on_image_array(
  #image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
  #np.squeeze(scores), category_index, use_normalized_coordinates=True,
  #min_score_thresh=.9,
  #line_thickness=4)
  
  return image_np
  
# Define function for handling images
def detect_image(image_path, sess, detection_graph):
  #Import image
  image = cv2.imread(image_path)
  image = cv2.resize(image, (width, height))
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  #Detect objects
  image_np = detect_objects(image_np, sess, detection_graph)

  #cv2.imwrite(output, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.waitKey(0)

def detect_image_webcam(image, sess, detection_graph):
  # Format data
  image = cv2.resize(image, (width, height))
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  # Detect objects
  image_np = detect_objects(image_np, sess, detection_graph)

  #cv2.imwrite(output, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.waitKey(1)
  return image_np

def detect_objects_coords(image_np, sess, detection_graph):
    # Define input
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    # Define outputs
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    
    # Predict
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    # Find box vertices
    box_coords = []
    for i in range(0, len(scores[0])):
        if scores[0][i] > .84:
            box = boxes[0][i]
            rows = image_np.shape[0]
            cols = image_np.shape[1]
            box[0] = box[0]
            box[1] = box[1]
            box[2] = box[2]
            box[3] = box[3]
            #box[0] = box[0]*rows
            #box[1] = box[1]*cols
            #box[2] = box[2]*rows
            #box[3] = box[3]*cols
 
            box_coords.append(box)
            #cv2.rectangle(image_np, (box[1], box[0]), (box[3], box[2]), (0,255,0),3)
    #cv2.line(img, (0, 242), (height, 242), (255,0,0), thickness=3)
    #cv2.line(img, (0, 300), (height, 300), (255,0,0), thickness=3)
    #cv2.line(image_np, (242, 0), (242, height), (255,0,0), thickness=3)
    #cv2.line(image_np, (300, 0), (300, height), (255,0,0), thickness=3)
    
    #cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(1)
    
    # Returns coords of box in [y1, x1, y2, x2] format
    return box_coords
    

def detect_image_coords(image_path, sess, detection_graph):
            #Import image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (width, height))
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Detect objects
            coords = detect_objects_coords(image_np, sess, detection_graph)
            return coords
import sys

################################################################################
# PiCamera to Gambezi adapter
class GambeziOutput(object):

    ############################################################
    def __init__(self, gn_stream, gn_stream_fps):
        self.gn_stream = gn_stream
        self.gn_stream_fps = gn_stream_fps
        self.old_time = time.time()
        self.period = 1/30

    ############################################################
    def write(self, b):
        # Measure FPS
        new_time = time.time()
        period = new_time - self.old_time
        self.old_time = new_time
        self.period = 0.99 * self.period + (1 - 0.99) * period
        # Write to gambezi
        self.gn_stream.set_data(b, 0, len(b))
        self.gn_stream_fps.set_double(1/self.period)

    ############################################################
    def flush(self):
        pass

################################################################################
# PiCamera RGB Analysis adapter
class RGBAnalysis(PiRGBAnalysis):

    ############################################################
    def __init__(self, camera, gn_count, gn_x1, gn_y1, gn_x2, gn_y2):
        super(RGBAnalysis, self).__init__(camera)
        self.gn_count = gn_count
        self.gn_x1 = gn_x1
        self.gn_y1 = gn_y1
        self.gn_x2 = gn_x2
        self.gn_y2 = gn_y2
        self.counter = 0

    ############################################################
    def analyse(self, image):
        # Refresh view 1 times/s
        self.counter += 1
        if self.counter >= 30 / 30:
            self.counter = 0
            interval = True
        else:
            interval = False

        if interval:
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Detect objects
            coords = detect_objects_coords(image_np, sess, detection_graph)
            print(coords)
            gn_count.set_double(len(coords))
            for i in range(0, len(coords)):
                gn_x1.get_node(str(i)).set_double(float(coords[i][0]))
                gn_y1.get_node(str(i)).set_double(float(coords[i][1]))
                gn_x2.get_node(str(i)).set_double(float(coords[i][2]))
                gn_y2.get_node(str(i)).set_double(float(coords[i][3]))

# Detect images
if (__name__ == '__main__'):
    # Start Session
    with detection_graph.as_default():
    	sess = tf.Session(graph=detection_graph, config=tf.ConfigProto(intra_op_parallelism_threads=8))
    
    # Initialize Gambezi
    gambezi = Gambezi('10.17.47.2:5809', True)
    gn_count = gambezi.get_node('pi_vision/count')
    gn_x1 = gambezi.get_node('pi_vision/x1')
    gn_y1 = gambezi.get_node('pi_vision/y1')
    gn_x2 = gambezi.get_node('pi_vision/x2')
    gn_y2 = gambezi.get_node('pi_vision/y2')

    # Initialize local Gambezi
    gambezi2 = Gambezi('localhost:5809', True)
    gambezi2.get_node('pi_vision/width').set_double(width)
    gambezi2.get_node('pi_vision/height').set_double(height)
    gn_stream = gambezi2.get_node('pi_vision/stream')
    gn_stream_fps = gambezi2.get_node('pi_vision/framerate')

    # Loop through webcam frames
    cam = PiCamera()
    cam.awb_mode = 'off'
    cam.awb_gains = (1.15, 2.46)
    cam.exposure_mode = 'auto'
    cam.framerate = 30
    cam.rotation = 270
    cam.shutter_speed = 10000
    cam.resolution = (width, height)
    time.sleep(1)
        
    ################################################################################
    print('Starting stream')
    gambezi_output = GambeziOutput(gn_stream, gn_stream_fps)
    cam.start_recording(gambezi_output, format='h264', splitter_port=1, profile="baseline", level='4', bitrate=1000000, quality=30)

    ################################################################################
    print('Starting vision processing')
    analysis_output = RGBAnalysis(cam, gn_count, gn_x1, gn_y1, gn_x2, gn_y2)
    cam.start_recording(analysis_output, 'bgr', splitter_port=2)
