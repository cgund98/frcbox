import numpy as np
import os
import tensorflow as tf
import cv2
from imutils.video import FPS
import sys

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
    #print(scores[0][:4])
    for i in range(0, len(scores[0])):
        if scores[0][i] > .4:
            box = boxes[0][i]
            rows = image_np.shape[0]
            cols = image_np.shape[1]
            #box[0] = box[0]
            #box[1] = box[1]
            #box[2] = box[2]
            #box[3] = box[3]
            box[0] = box[0]*rows
            box[1] = box[1]*cols
            box[2] = box[2]*rows
            box[3] = box[3]*cols
            box_coords.append(box)
    #cv2.line(img, (0, 242), (height, 242), (255,0,0), thickness=3)
    #cv2.line(img, (0, 300), (height, 300), (255,0,0), thickness=3)
    #cv2.line(image_np, (242, 0), (242, height), (255,0,0), thickness=3)
    #cv2.line(image_np, (300, 0), (300, height), (255,0,0), thickness=3)
    
    
    # Returns coords of box in [y1, x1, y2, x2] format
    return box_coords
    

def detect_image(image_path, sess, detection_graph):
    #Import image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (960, 1280))
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Detect objects
    box_coords = detect_objects_coords(image_np, sess, detection_graph)
    for box in box_coords:
        cv2.rectangle(image_np, (box[1], box[0]), (box[3], box[2]), (0,255,0),3)
    
    print("Showing predictions...")
    cv2.imshow('img', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    
    return box_coords

# Detect objects in a video
def detect_video(video_path, sess, detection_graph):
    cap = cv2.VideoCapture(video_path)
    width, height = cap.get(3), cap.get(4)
    resize_factor = 1000 / width
    #tracker = cv2.TrackerCSRT_create()
    tracker = cv2.TrackerKCF_create()
    b_box = None # Bounding box of biggest object
    fps = FPS().start()
    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        image = frame
        image = cv2.resize(image, (int(width * resize_factor), int(height * resize_factor)))
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        i += 1

        #Detect objects
        if b_box is None or i % 500 == 0: # Every 500 frames, use detection again
            box_coords = detect_objects_coords(image_np, sess, detection_graph)
            widths = [x[1]-x[0] for x in box_coords]
            b_box = box_coords[widths.index(max(widths))]
            #cv2.rectangle(image, (b_box[1], b_box[0]), (b_box[3], b_box[2]), (0,255,0),3)
            b_box = (b_box[1], b_box[0], b_box[3]-b_box[1], b_box[2]-b_box[0]) # Change bounding box to (x, y, w, h) format
            #b_box = cv2.selectROI("img", image, fromCenter=False, showCrosshair=True) # Manual bounding box
            
            tracker.init(image, b_box)
    
            #return
            #for box in box_coords:
            #    cv2.rectangle(image, (box[1], box[0]), (box[3], box[2]), (0,255,0),3)
        else: # Detect objects with tracker, toggle comments to test fps with dl
            (success, box) = tracker.update(image) # box in format [x, y, w, h]
            #success = False

            if success:
                (x, y, w, h) = [int(x) for x in box]
                cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 3)
            
            #box_coords = detect_objects_coords(image_np, sess, detection_graph)
            #widths = [x[1]-x[0] for x in box_coords]
            #b_box = box_coords[widths.index(max(widths))]
            #cv2.rectangle(image, (b_box[1], b_box[0]), (b_box[3], b_box[2]), (0,255,0),3)

            fps.update()
            fps.stop()

            cv2.putText(image, "FPS: {:.2f}".format(fps.fps()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        #print("Showing predictions...")
        cv2.imshow('img', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Detect images
def main():
    # For reading from image files
    #input_dir = '../frcbox_test_video/images/'
    #input_dir = '../frcbox_test_images/images/'
    #image_paths = sorted([ input_dir + f for f in os.listdir(input_dir)])

    if len(sys.argv) == 1: print("No file given as argument."); return
    inp = sys.argv[1]
    img_file_types = ["png", "jpg"]
    video_file_types = ["mov", "mp4"]

    if inp[-3:] not in img_file_types and inp[-3:] not in video_file_types: print("File type not supported."); return

    # Start Session
    with detection_graph.as_default():
    	sess = tf.Session(graph=detection_graph)
    
    # Test Coord output
    #print(detect_image_coords(image_paths[5], sess, detection_graph))
    if inp[-3:] in img_file_types: detect_image(inp, sess, detection_graph)
    if inp[-3:] in video_file_types: detect_video(inp, sess, detection_graph)
    

if __name__ == '__main__':
    main();	
