# write opencv read 2 videos and compare them
from tqdm.auto import tqdm

import cv2
import numpy as np
import os
import sys
import time
from demo.default_config import config
import numpy as np

from estimators.gaze import GazeEstimator
from models.gaze_det.ptgaze.common import Face
from models.gaze_det.ptgaze.utils import generate_dummy_camera_params
from sklearn.metrics.pairwise import paired_euclidean_distances, paired_cosine_distances

if config.gaze_estimator.use_dummy_camera_params:
    generate_dummy_camera_params(config)

e = GazeEstimator.from_name(
    det_name="retinaface",
    align_name="fan",
    face3d_name="Face3DModel",
    gaze_name="GazeModel",
    cfg=config,
)


def compare_faces(face_1: Face, face_2: Face):
    bbox1 = face_1.bbox.reshape((1, 4))
    bbox2 = face_2.bbox.reshape((1, 4))
    landmarks1 = face_1.landmarks
    landmarks2 = face_2.landmarks
    gaze1 = face_1.gaze_vector.reshape((1, 3))
    gaze2 = face_2.gaze_vector.reshape((1, 3))

    euler_angles1 = face_1.head_pose_rot.as_euler("XYZ", degrees=True).reshape((1, 3))
    euler_angles2 = face_2.head_pose_rot.as_euler("XYZ", degrees=True).reshape((1, 3))

    return {
        "bbox L2 error": paired_euclidean_distances(bbox1, bbox2).mean(),
        "landmakrs L2 error": paired_euclidean_distances(landmarks1, landmarks2).mean(),
        "gaze L2 error": paired_euclidean_distances(gaze1, gaze2).mean(),
        "head posee angle error": paired_cosine_distances(
            euler_angles1, euler_angles2
        ).mean(),
    }


def compare(e, frame1, frame2):
    face1 = calc(e, frame1)
    face2 = calc(e, frame2)
    return compare_faces(face1, face2)


def calc(estimator, image):
    undistorted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undistorted = cv2.undistort(
        undistorted, estimator.camera.camera_matrix, estimator.camera.dist_coefficients,
    )
    undistorted = [undistorted]

    face = estimator.detect_faces(undistorted)[0]


    estimator.estimate_gaze(undistorted, face)
    return face


class AvgMeter:
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, item):
        for key, value in item.items():
            if key not in self.data:
                self.data[key] = 0
            self.data[key] += value

        self.sample_size += 1

    def value(self):
        for keys in self.data:
            self.result[keys] = self.data[keys] / self.sample_size
        return self.result

    def reset(self):
        self.data = {}
        self.result = {}
        self.sample_size = 0

    def summary(self):
        results = self.value()
        print(f"Evaluate {self.sample_size} frames")
        for key, value in results.items():
            print(f"{key} : {value:.6f}")


if __name__ == "__main__":
    # get video file path
    video_path = sys.argv[1]
    video_path2 = sys.argv[2]
    # get video file name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name2 = os.path.splitext(os.path.basename(video_path2))[0]
    # get video object
    cap = cv2.VideoCapture(video_path)
    cap2 = cv2.VideoCapture(video_path2)
    # get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps2 = cap2.get(cv2.CAP_PROP_FPS)
    # get video width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get video length
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    length2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    # get video duration
    duration = length / fps
    duration2 = length2 / fps2
    # check if video is open
    if not cap.isOpened() or not cap2.isOpened():
        print("Error opening video stream or file")
        sys.exit(1)
    meter = AvgMeter()
    total = min(length, length2)
    pbar = tqdm(total=total)
    # write tqdm progress bar for while

    while True:
        # get current frame
        ret, frame = cap.read()
        ret2, frame2 = cap2.read()
        # check if frame is read
        if not ret or not ret2:
            break
        try:
            res = compare(e, frame, frame2)
        except:
            pbar.update(1)
            continue
        meter.update(res)
        pbar.update(1)
    # release video object
    if ret == True:
        cap.release()

    if ret2 == True:
        cap2.release()

    cv2.destroyAllWindows()

    meter.summary()
