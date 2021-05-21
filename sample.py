import cv2
import numpy as np

from time import time
import datetime
from detector import MotionDetector
from packer import pack_images
from numba import jit
from picamera.array import PiRGBArray
from picamera import PiCamera

@jit(nopython=True)
def filter_fun(b):
    return ((b[2] - b[0]) * (b[3] - b[1])) > 300


if __name__ == "__main__":

    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    # cap = PiCamera()
    # cap.resolution = (1280, 720)
    # cap.framerate = 30

    detector = MotionDetector(bg_history=10,
                              bg_skip_frames=1,
                              movement_frames_history=2,
                              brightness_discard_level=5,
                              bg_subs_scale_percent=0.2,
                              pixel_compression_ratio=0.1,
                              group_boxes=True,
                              expansion_step=5)

    # group_boxes=True can be used if one wants to get less boxes, which include all overlapping boxes

    b_height = 320
    b_width = 320

    res = []
    fc = dict()
    ctr = 0
    # used to record the time when we processed last frame
    prev_frame_time = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break

        begin = time()

        boxes, frame = detector.detect(frame)
        # boxes hold all boxes around motion parts

        ## this code cuts motion areas from initial image and
        ## fills "bins" of 320x320 with such motion areas.
        ##
        results = []
        if boxes:
             results, box_map = pack_images(frame=frame, boxes=boxes, width=b_width, height=b_height,
                                            box_filter=filter_fun)
            # box_map holds list of mapping between image placement in packed bins and original boxes

        ## end

        for b in boxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 1)

        end = time()
        it = (end - begin) * 1000

        res.append(it)
        print("StdDev: %.4f" % np.std(res), "Mean: %.4f" % np.mean(res), "Last: %.4f" % it,
              "Boxes found: ", len(boxes))

        if len(res) > 10000:
            res = []

        # idx = 0
        # for r in results:
        #      idx += 1
        #      cv2.imshow('packed_frame_%d' % idx, r)

        ctr += 1
        nc = len(results)
        if nc in fc:
            fc[nc] += 1
        else:
            fc[nc] = 0

        if ctr % 100 == 0:
            print("Total Frames: ", ctr, "Packed Frames:", fc)

        # time when we finish processing for this frame

        # Calculating the fps

        # fps will be number of frame processed in given time frame
        # since their will be most of time error of 0.001 second
        # we will be subtracting it to get more accurate result
        fps = 1 / (begin - prev_frame_time)
        prev_frame_time = begin

        # converting the fps into integer
        fps = int(fps)

        # converting the fps to string so that we can display it on frame
        # by using putText function

        display_text = str(fps)

        # Use putText() method for
        # inserting text on video
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    display_text,
                    (20, 50),
                    font, 1,
                    (0, 255, 255),
                    2,
                    cv2.LINE_4)
        cv2.imshow('last_frame', frame)
        #cv2.imshow('detect_frame', detector.detection_boxed)
        #cv2.imshow('diff_frame', detector.color_movement)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(fc, ctr)
