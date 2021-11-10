"""
"""

from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn

import numpy as np


from models.gaze_det.base import BaseModel

from models.gaze_det.ptgaze.common import Camera, Face, FacePartsName
from models.gaze_det.ptgaze.head_pose_estimation import (
    HeadPoseNormalizer,
    LandmarkEstimator,
)
from models.gaze_det.ptgaze.models import create_model
from models.gaze_det.ptgaze.transforms import create_transform
from models.gaze_det.ptgaze.utils import get_3d_face_model


class Face3DModel(BaseModel):
    def __init__(self, config: DictConfig):
        super(Face3DModel, self).__init__()
        self._config = config
        assert (
            config.mode == "MPIIGaze" or config.mode == "ETH-XGaze"
        ), "Only ETH-XGaze and MPIIGaze are supported"
        self._face_model3d = get_3d_face_model(config)
        self.camera = Camera(config.gaze_estimator.camera_params)
        self._normalized_camera = Camera(config.gaze_estimator.normalized_camera_params)

        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera,
            self._normalized_camera,
            self._config.gaze_estimator.normalized_camera_distance,
        )

    def preprocess(self, bbox, landmarks):
        return Face(bbox, landmarks)

    def forward(self, image, face: Face):
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face, self._config.mode)
        self._head_pose_normalizer.normalize(image, face)
        return face

    def postprocess(self, prediction, face):
        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()
        return face


if __name__ == "__main__":
    test_params = {
        "mode": "ETH-XGaze",
        "model": {"name": "resnet18"},
        "face_detector": {"mode": "Face68",},
        "gaze_estimator": {
            "checkpoint": "/home/nhtlong/.ptgaze/models/eth-xgaze_resnet18.pth",
            "camera_params": "/tmp/camera_params.yaml",
            "use_dummy_camera_params": True,
            "normalized_camera_params": "/home/nhtlong/workspace/mediaeval21/dr-ws/demo/data/normalized_camera_params/eth-xgaze.yaml",
            "normalized_camera_distance": 0.6,
            "image_size": [224, 224],
        },
        "PACKAGE_ROOT": "/home/nhtlong/workspace/mediaeval21/dr-ws/demo",
    }
    config = DictConfig(test_params)
    model = Face3DModel(config)
