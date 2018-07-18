
import numpy as np
import os
import six.moves.urllib as urllib
import sys  
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict

from PIL import Image
sys.path.append("..")
from object_detection.utils import ops as utils_ops
import cv2

from utils import label_map_util

from utils import visualization_utils as vis_util


MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

detection_graph = tf.Graph()
  
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (12, 8)
# print(categories)
cap=cv2.VideoCapture(0)  

with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')
	with tf.Session() as sess:
		while True:

			ret, image=cap.read()
						
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
				'num_detections', 'detection_boxes', 'detection_scores',
				'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						tensor_name)
			if 'detection_masks' in tensor_dict:
				
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
		
			# Run inference
			output_dict = sess.run(tensor_dict,
									feed_dict={image_tensor: np.expand_dims(image, 0)})

			
			num_detections= int(output_dict['num_detections'][0])
			detection_classes= output_dict[
				'detection_classes'][0].astype(np.uint8)
			# print(detection_classes)
			# engine.say('Good morning.')
			# engine.runAndWait()
			
			detection_boxes= output_dict['detection_boxes'][0]
			detection_scores= output_dict['detection_scores'][0]
			if 'detection_masks' in globals() and (not detection_masks==None):
				detection_masks = detection_masks[0]
			else:
				detection_masks = None
  
			vis_util.visualize_boxes_and_labels_on_image_array(
				image,
				detection_boxes,
				detection_classes,
				detection_scores,
				category_index,
				instance_masks=detection_masks,
				use_normalized_coordinates=True,
				line_thickness=6)
			cv2.imshow('object detection', image)
			if cv2.waitKey(25) & 0xFF== ord("q"):
				cv2.destroyAllWindows()
				break


