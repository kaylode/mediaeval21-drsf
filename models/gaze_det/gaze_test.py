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
from .face3d import Face3DModel

class GazeModelTest(BaseModel):
    def __init__(self, config: DictConfig, loss_fn: str = "l2"):
        super(GazeModelTest, self).__init__()
        self._config = config
        assert (
            config.mode == "MPIIGaze" or config.mode == "ETH-XGaze"
        ), "Only ETH-XGaze and MPIIGaze are supported"
        self._gaze_estimation_model = self._load_model()
        self._face3d = Face3DModel(config)
        self._transform = create_transform(config)

        if loss_fn == "l2":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function does not exist")

    def preprocess(self, images, boxes, landmarks):
        face_ls = []
        for bbox, lm in zip(boxes, landmarks):
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float,)
            face3d = self._face3d.preprocess(bbox, lm)
            face_ls.append(face3d)

        batch_inputs = []
        for image, face in zip(images, face_ls):
            face = self._face3d.forward(image, face)[0]
            input_image = self._transform(face.normalized_image).unsqueeze(0)
            batch_inputs.append(input_image)

        batch_inputs = torch.stack(batch_inputs, dim=0)
        return batch_inputs

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

    def detect(self, x):
        with torch.no_grad():
            preds = self._gaze_estimation_model.forward(x)
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

