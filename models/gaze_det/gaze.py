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

