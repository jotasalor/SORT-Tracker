'''
Implement and test car detection (localization)
'''

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt
import time
from glob import glob
cwd = os.path.dirname(os.path.realpath(__file__))


import core.utils as utils


class CarDetector(object):
    def __init__(self):

        self.car_boxes = []

        os.chdir(cwd)

        self.return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0",
                           "pred_lbbox/concat_2:0"]
        self.pb_file = "./yolov3/frozen_inference_graph.pb"
        self.num_classes = 80
        self.input_size = 416
        self.detection_graph = tf.Graph()

        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.detection_graph, config=config)
        self.return_tensors = utils.read_pb_return_tensors(self.detection_graph, self.pb_file, self.return_elements)


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

        original_image_size = image.shape[:2]
        image_data = utils.image_preporcess(np.copy(image), [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]

        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
                [self.return_tensors[1], self.return_tensors[2], self.return_tensors[3]],
                feed_dict={self.return_tensors[0]: image_data})

        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)

        bboxes, tboxes, classes, scores = utils.postprocess_boxes(pred_bbox, original_image_size, self.input_size, 0.3)
        bboxes = utils.nms(bboxes, 0.45, method='nms')

        """""
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        --->    [y_min, x_min, y_max, x_max] [] []
        """""
        i = [1, 0, 3, 2] #permutation order
        dim = image.shape[0:2]
        # boxes = tboxes[:, i]
        # boxes[:,[0,2]] = boxes[:,[0,2]]/dim[0]
        # boxes[:,[1,3]] = boxes[:,[1,3]]/dim[1]


        boxes = []
        classes = []
        scores = []
        for box in bboxes:
            tbox = [box[1]/dim[0], box[0]/dim[1], box[3]/dim[0], box[2]/dim[1]]
            tclass = int(box[5])
            tscore = box[4]
            boxes.append(tbox)
            classes.append(tclass)
            scores.append(tscore)

        # boxes = np.squeeze(boxes)
        # classes = np.squeeze(classes)
        # scores = np.squeeze(scores)

        #cls = classes.tolist()
        cls = classes

        # Filter low confidence objects
        idx_vec = [i for i, v in enumerate(cls) if (scores[i] > 0.3)]

        if len(idx_vec) == 0:
            print('no detection!')
            self.car_boxes = []
        else:
            tmp_car_boxes = []
            for idx in idx_vec:
                dim = image.shape[0:2]
                box = self.box_normal_to_pixel(boxes[idx], dim)
                box_h = box[2] - box[0]
                box_w = box[3] - box[1]
                ratio = box_h / (box_w + 0.01)

#                if ((ratio < 0.8) and (box_h > 20) and (box_w > 20)):
                tmp_car_boxes.append(box)
                print(box, ', confidence: ', scores[idx], 'ratio:', ratio)

 #               else:
 #                  print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)

            self.car_boxes = tmp_car_boxes

        return self.car_boxes


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

