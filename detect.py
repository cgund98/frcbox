import numpy as np
import os
import tensorflow as tf
import cv2

# Import utilities
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Define import paths
PATH_TO_CKPT = 'output_inference_graph.pb/frozen_inference_graph.pb' # Import frozen model
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
  vis_util.visualize_boxes_and_labels_on_image_array(
  image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
  np.squeeze(scores), category_index, use_normalized_coordinates=True,
  min_score_thresh=.9,
  line_thickness=4)
  
  return image_np
  
# Define function for handling images
def detect_image(image_path, sess, detection_graph):
  #Import image
  image = cv2.imread(image_path)
  image = cv2.resize(image, (1080, 1920))
  image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  
  #Detect objects
  image_np = detect_objects(image_np, sess, detection_graph)

  #cv2.imwrite(output, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
  cv2.waitKey(0)

def detect_image_webcam(image, sess, detection_graph):
  # Format data
  image = cv2.resize(image, (480, 640))
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
    box = boxes[0][0]
    rows = image_np.shape[0]
    cols = image_np.shape[1]
    box[0] = box[0]*rows
    box[1] = box[1]*cols
    box[2] = box[2]*rows
    box[3] = box[3]*cols
    #cv2.rectangle(image_np, (box[1], box[0]), (box[3], box[2]), (0,255,0),3)
    #cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    #cv2.waitKey(0)
    
    # Returns coords of box in [y1, x1, y2, x2] format
    return box
    

def detect_image_coords(image_path, sess, detection_graph):
            #Import image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (480, 640))
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #Detect objects
            coords = detect_objects_coords(image_np, sess, detection_graph)
            return coords
import sys
# Detect images
if __name__ == '__main__':
	# For reading from image files
	#input_dir = '../frcbox_test_video/images/'
	#input_dir = '../frcbox_test_images/images/'
	#image_paths = sorted([ input_dir + f for f in os.listdir(input_dir)])

	inp = sys.argv[1]
	# Start Session
	with detection_graph.as_default():
		sess = tf.Session(graph=detection_graph)
	
	# Test Coord output
	#print(detect_image_coords(image_paths[5], sess, detection_graph))
	detect_image(inp, sess, detection_graph)
	# Loop through images and detect boxes (for image files)
	#for i in range(0, len(image_paths)):
	#	detect_image(image_paths[i], sess, detection_graph)
	
	# Loop through webcam frames	
	#cam = cv2.VideoCapture(0)
	#while True:
	#	frame = cam.read()
	#frame_np = detect_image_webcam(frame, sess, detection_graph)
	#	
	#	if cv2.waitKey(1) & 0xFF == ord('q'):
	#		break
		
