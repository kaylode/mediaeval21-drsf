"""
"""

from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn

import numpy as np


from models.gaze_det.base import BaseModel


from models.gaze_det.base import BaseModel

from models.gaze_det.ptgaze.common import Camera, Face, FacePartsName
from models.gaze_det.ptgaze.head_pose_estimation import (
    HeadPoseNormalizer,
    LandmarkEstimator,
)
from models.gaze_det.ptgaze.models import create_model
from models.gaze_det.ptgaze.transforms import create_transform
from models.gaze_det.ptgaze.utils import get_3d_face_model


class GazeModel(BaseModel):
    def __init__(self, config: DictConfig):
        super(GazeModel, self).__init__()
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
        self._gaze_estimation_model = self._load_model()
        self._transform = create_transform(config)

    def preprocess(self, face: Face):
        input_image = self._transform(face.normalized_image).unsqueeze(0)
        return input_image

    def postprocess(self, prediction, face):
        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

    def forward(self, images, targets=None):

        preds = self._gaze_estimation_model.forward(images)
        # Regression loss
        if targets is not None:
            loss = self.loss_fn(preds, targets)
        return loss

    def detect(self, image):
        preds = self._gaze_estimation_model.forward(image)
        return preds.cpu().detach().numpy()

    def make_targets(self, predictions):
        return torch.from_numpy(predictions[0]).to(self.device)

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(
            self._config.gaze_estimator.checkpoint, map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
        model.to(torch.device(self._config.device))
        model.eval()
        return model


if __name__ == "__main__":
    test_params = {
        "mode": "ETH-XGaze",
        "device": "cpu",
        "model": {"name": "resnet18"},
        "face_detector": {
            "mode": "mediapipe",
            "dlib_model_path": "/home/nhtlong/.ptgaze/dlib/shape_predictor_68_face_landmarks.dat",
            "mediapipe_max_num_faces": 3,
        },
        "gaze_estimator": {
            "checkpoint": "/home/nhtlong/.ptgaze/models/eth-xgaze_resnet18.pth",
            "camera_params": "/tmp/camera_params.yaml",
            "use_dummy_camera_params": True,
            "normalized_camera_params": "/home/nhtlong/workspace/mediaeval21/dr-ws/demo/data/normalized_camera_params/eth-xgaze.yaml",
            "normalized_camera_distance": 0.6,
            "image_size": [224, 224],
        },
        "demo": {
            "use_camera": False,
            "display_on_screen": False,
            "wait_time": 1,
            "image_path": None,
            "video_path": "../assets/T002_ActionsShorter_mini_8829_9061_Talk-non-cell.mp4",
            "output_dir": ".",
            "output_file_extension": "avi",
            "head_pose_axis_length": 0.05,
            "gaze_visualization_length": 0.05,
            "show_bbox": True,
            "show_head_pose": True,
            "show_landmarks": True,
            "show_normalized_image": False,
            "show_template_model": True,
        },
        "PACKAGE_ROOT": "/home/nhtlong/workspace/mediaeval21/dr-ws/demo",
    }
    config = DictConfig(test_params)
    model = GazeModel(config)
