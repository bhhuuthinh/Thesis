import numpy as np
import cv2

class Video:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)

    def next_frame(self):
        ret, frame = self.cap.read()
        if ret == True:
            self.current_frame = frame
            return frame
        else:
            return None

    def get_tag(self):
        gray = cv2.cvtColor(self.current_frame,1)

        width = gray.shape[1]
        height = gray.shape[0]
        t_width = (int)(231.0 / 1920.0 * width)
        t_height = (int)(45.0 / 1080.0 * height)

        left_1 = (int)(167.0 / 1920.0 * width)
        top_1 = (int)(959.0 / 1080.0 * height)
        left_tag = gray[top_1:top_1 + t_height, left_1:left_1 + t_width,:]

        left_2 = (int)(1483.0 / 1920.0 * width)
        top_2 = (int)(959.0 / 1080.0 * height)
        right_tag = gray[top_2:top_2 + t_height, left_2:left_2 + t_width,:]

        return left_tag, right_tag

    def release(self):
        self.cap.release()