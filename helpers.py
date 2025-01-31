#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Helper classes and functions for detection and tracking
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors

class Box:
    def __init__(self):
        self.x, self.y = float(), float()
        self.w, self.h = float(), float()
        self.c = float()
        self.prob = float()

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1 / 2.;
    l2 = x2 - w2 / 2.;
    left = max(l1, l2)
    r1 = x1 + w1 / 2.;
    r2 = x2 + w2 / 2.;
    right = min(r1, r2)
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if w < 0 or h < 0: return 0;
    area = w * h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w * a.h + b.w * b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b);

def box_iou2(a, b):
    '''
    Helper funciton to calculate the ratio between intersection and the union of
    two boxes a and b
    a[0], a[1], a[2], a[3] <-> left, up, right, bottom
    '''
    
    w_intsec = np.maximum (0, (np.minimum(a[2], b[2]) - np.maximum(a[0], b[0])))
    h_intsec = np.maximum (0, (np.minimum(a[3], b[3]) - np.maximum(a[1], b[1])))
    s_intsec = w_intsec * h_intsec
    s_a = (a[2] - a[0])*(a[3] - a[1])
    s_b = (b[2] - b[0])*(b[3] - b[1])
  
    return float(s_intsec)/(s_a + s_b -s_intsec)

def convert_to_pixel(box_yolo, img, crop_range):
    '''
    Helper function to convert (scaled) coordinates of a bounding box 
    to pixel coordinates. 
    
    Example (0.89361443264143803, 0.4880486045564924, 0.23544462956491041, 
    0.36866588651069609)
    
    crop_range: specifies the part of image to be cropped
    '''
    
    box = box_yolo
    imgcv = img
    [xmin, xmax] = crop_range[0]
    [ymin, ymax] = crop_range[1]
    h, w, _ = imgcv.shape
    
    # Calculate left, top, width, and height of the bounding box
    left = int((box.x - box.w/2.)*(xmax - xmin) + xmin)
    top = int((box.y - box.h/2.)*(ymax - ymin) + ymin)
    
    width = int(box.w*(xmax - xmin))
    height = int(box.h*(ymax - ymin))
    
    # Deal with corner cases
    if left  < 0    :  left = 0
    if top   < 0    :   top = 0
    
    # Return the coordinates (in the unit of the pixels)
  
    box_pixel = np.array([left, top, width, height])
    return box_pixel



def convert_to_cv2bbox(bbox, img_dim = (1280, 720)):
    '''
    Helper fucntion for converting bbox to bbox_cv2
    bbox = [left, top, width, height]
    bbox_cv2 = [left, top, right, bottom]
    img_dim: dimension of the image, img_dim[0]<-> x
    img_dim[1]<-> y
    '''
    left = np.maximum(0, bbox[0])
    top = np.maximum(0, bbox[1])
    right = np.minimum(img_dim[0], bbox[0] + bbox[2])
    bottom = np.minimum(img_dim[1], bbox[1] + bbox[3])
    
    return (left, top, right, bottom)
    
    
def draw_box_label(img, bbox_cv2, box_color, show_label=True):
    '''
    Helper funciton for drawing the bounding boxes and the labels
    bbox_cv2 = [left, top, right, bottom]
    '''
    #box_color= (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    font_color = (0, 0, 0)
    left, top, right, bottom = bbox_cv2[1], bbox_cv2[0], bbox_cv2[3], bbox_cv2[2]
    
    # Draw the bounding box
    cv2.rectangle(img, (left, top), (right, bottom), box_color, 4)
    
    if show_label:
        # Draw a filled box on top of the bounding box (as the background for the labels)
        cv2.rectangle(img, (left-2, top-45), (right+2, top), box_color, -1, 1)
        
        # Output the labels that show the x and y coordinates of the bounding box center.
        text_x= 'x='+str((left+right)/2)
        cv2.putText(img,text_x,(left,top-25), font, font_size, font_color, 1, cv2.LINE_AA)
        text_y= 'y='+str((top+bottom)/2)
        cv2.putText(img,text_y,(left,top-5), font, font_size, font_color, 1, cv2.LINE_AA)
    
    return img    

def trk_id_to_color(id):
    colormap = {"A": (255, 255, 120), "B": (255, 255, 80), "C": (255, 255, 0), "D": (255, 25, 255),
                "E": (255, 0, 78), "F": (89, 210, 37), "G": (92, 2, 27), "H": (0, 255, 0),
                "I": (0, 0, 255), "J": (130, 255, 9), "K": (176, 0, 255), "L": (255, 36, 0),
                "M": (58, 54, 58), "N": (4, 236, 42), "O": (91, 83, 90), "P": (59, 182, 78),
                "Q": (86, 54, 89), "R": (0, 0, 79), "S": (0, 80, 0), "T": (50, 11, 99),
                "U": (255, 23, 78), "V": (0, 67, 1), "W": (255, 0, 127), "X": (252, 0, 120),
                "Y": (94, 83, 52), "Z": (13, 59, 214),
                "AA": (127, 127, 60), "AB": (127, 127, 40), "AC": (127, 127, 0), "AD": (127, 12, 127),
                "AE": (127, 0, 40), "AF": (45, 105, 20), "AG": (45, 1, 15), "AH": (0, 127, 0),
                "AI": (0, 0, 127), "AJ": (65, 127, 5), "AK": (90, 0, 127), "AL": (127, 20, 0),
                "AM": (30, 30, 30), "AN": (2, 120, 21), "AO": (45, 41, 45), "AP": (30, 91, 39),
                "AQ": (43, 27, 43), "AR": (0, 0, 40), "AS": (0, 40, 0), "AT": (25, 6, 50),
                "AU": (127, 12, 38), "AV": (0, 34, 1), "AW": (127, 0, 68), "AX": (123, 0, 60),
                "AY": (47, 41, 26), "AZ": (7, 30, 122)
                }
    color = colormap[id]

    return color
