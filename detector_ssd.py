"""
SCRIPT SSD Detector Implementation
Jorge Sánchez-Alor Expósito
Based on code from @author: Kyle Guan
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))

# Tensorflow visualization_unitls
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

class CarDetector(object):
    def __init__(self):

        self.car_boxes = []

        os.chdir(cwd)

        #Tensorflow localization/detection model
        # Single-shot-dectection with mobile net architecture

        #detect_model_name = 'ssd_mobilenet_v1_coco_11_06_2017'
        #detect_model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
        detect_model_name = 'ssd_mobilenet_v2_bdd100k'

        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'

        # setup tensorflow graph
        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')

####################################################
#            for op in self.detection_graph.get_operations():
#                print(op.name)
####################################################

            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)

    def get_localization(self, image, visual=False):

        """Determines the locations of the cars in the image

        Args:
            image: camera image

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

        """
        #PATH_TO_LABELS = './mscoco_label_map.pbtxt'
        PATH_TO_LABELS = './bdd_label_map.pbtxt'

        # LOADING LABEL MAP
        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})

              if visual == True:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      line_thickness=3)

                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()

              boxes= np.squeeze(boxes)
              classes = np.squeeze(classes)
              scores = np.squeeze(scores)

              cls = classes.tolist()

              # Filter low confidence
              idx_vec = [i for i, v in enumerate(cls) if (scores[i]>0.6)]

              if len(idx_vec) ==0:
                  print('no detection!')
                  self.car_boxes = []
              else:
                  tmp_car_boxes = []
                  tmp_car_scores = []
                  tmp_car_classes = []
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      box = self.box_normal_to_pixel(boxes[idx], dim)
                      box_h = box[2] - box[0]
                      box_w = box[3] - box[1]
                      ratio = box_h/(box_w + 0.01)

#                      if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                      tmp_car_boxes.append(box)
                      tmp_car_classes.append(classes[idx])
                      tmp_car_scores.append(scores[idx])
                      print(box, ', confidence: ', scores[idx], 'ratio:', ratio)

#                      else:
#                          print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)



                  self.obj_boxes = tmp_car_boxes
                  self.obj_scores= tmp_car_scores
                  self.obj_classes = tmp_car_classes
        return self.obj_boxes, self.obj_scores, self.obj_classes

if __name__ == '__main__':
        # Test the performance of the detector
        det =CarDetector()
        os.chdir(cwd)
        TEST_IMAGE_PATHS= glob(os.path.join('test_images/', '*.jpg'))

        for i, image_path in enumerate(TEST_IMAGE_PATHS[0:2]):
            print('')
            print('*************************************************')

            img_full = Image.open(image_path)
            img_full_np = det.load_image_into_numpy_array(img_full)
            img_full_np_copy = np.copy(img_full_np)
            start = time.time()
            b = det.get_localization(img_full_np, visual=False)
            end = time.time()
            print('Localization time: ', end-start)
#

