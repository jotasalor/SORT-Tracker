"""
SCRIPT Detection & tracking Precision Evaluation
Jorge Sánchez-Alor Expósito
Based on code from @author: kyleguan
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import helpers
import detector
import tracker

# Global variables to be used by functions of VideoFileClip
frame_count = 0 # frame counter

max_age = 2  # no.of consecutive unmatched detection before
             # a track is deleted

min_hits =3  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                      '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26',
                      '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                      '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52',
                      '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65',
                      '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78'])

track_id_list = deque([str(_+1) for _ in range(2000)])

debug = False

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det)
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    


def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug


    frame_count+=1

    start = time.time()

    img_dim = (img.shape[1], img.shape[0])
    z_box, tmp_confidence, tmp_classes = det.get_localization(img) # measurement

    if debug:
       print('Frame:', frame_count)
       
    x_box =[]
    if debug: 
        for i in range(len(z_box)):
           img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
           plt.imshow(img1)
        plt.show()
    
    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
    
    
    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  
    if debug:
         print('Detection: ', z_box)
         print('x_box: ', x_box)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)
    
    confidence = []
    # Deal with matched detections     
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            confidence.append(tmp_confidence[det_idx])
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0
    
    # Deal with unmatched detections      
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            confidence.append(tmp_confidence[idx])
            x_box.append(xx)
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
                   
       
    # The list of tracks to be annotated  
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             img= helpers.draw_box_label(img, x_cv2, helpers.trk_id_to_color(trk.id)) # Draw the bounding boxes on the
                                             # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))


    frame_num = frame_count
    idx = 0
    for track in tracker_list:
        id = track.id
        bbox = track.box
        x1, y1, width, height = bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]
        conf = -1
        #conf = confidence[idx]
        #x, y, z = -1, -1, -1
        print('{},{},{},{},{},{},{},-1,-1,-1'.format(frame_num, id, x1, y1, width, height, conf),
              file=ht_file)
        idx +=1

    return img
    
if __name__ == "__main__":    
    
    det = detector.CarDetector()

    images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]

    ht_file = open('ht.txt', 'w+')

    codec = cv2.VideoWriter_fourcc(*'XVID')
    vid_fps = 5
    vid_width, vid_height = int(1280), int(720)

    out = cv2.VideoWriter('./data/results.avi', codec, vid_fps, (vid_width, vid_height))

    for i in range(len(images)):
         image = images[i]
         image_box = pipeline(image)
         plt.imshow(image_box)

         #plt.show()
         out.write(cv2.cvtColor(image_box, cv2.COLOR_BGR2RGB))

         if cv2.waitKey(1) == ord('q'):
            break

    out.release()
    cv2.destroyAllWindows()
    ht_file.close()
