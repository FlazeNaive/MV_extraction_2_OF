import os
import unittest
import time
import numpy as np
from mvextractor.videocap import VideoCap


cap = VideoCap()
filename = "dataset/alley_1.MPEG"
filename1 = "alley_1.MPEG"
filename2 = "vid_h264.mp4"
video_url = os.getenv('VIDEO_URL', filename2)

print(video_url)
print(type(video_url))

ret = cap.open(video_url)
ret = cap.open(filename1)

ret, frame, motion_vectors, frame_type, timestamp = cap.read()
print(motion_vectors)
ret, frame, motion_vectors, frame_type, timestamp = cap.read()
print(motion_vectors)
