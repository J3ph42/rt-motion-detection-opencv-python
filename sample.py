import cv2
from time import time
from detector import MovementDetector

if __name__ == "__main__":

    cap = cv2.VideoCapture('helmets-v1-55.mp4')

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    pixel_compression_ratio = 10

    detector = MovementDetector(pixel_compression_ratio=pixel_compression_ratio,
                                bg_history=20,
                                expansion_step=5)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        begin = time()
        detector.detect(frame)
        end = time()
        print(detector.count, 1000 * (end - begin), len(detector.bg_frames), len(detector.boxes))
        cv2.imshow('last_frame', detector.last_frame)
        cv2.imshow('detect_frame', detector.detection)
        cv2.imshow('diff_frame', detector.color_movement)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break