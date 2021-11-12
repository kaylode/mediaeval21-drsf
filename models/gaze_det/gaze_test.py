
from omegaconf.dictconfig import DictConfig
import torch
import torch.nn as nn
import numpy as np

from models.gaze_det.base import BaseModel
from models.gaze_det.ptgaze.models import create_model
from models.gaze_det.ptgaze.transforms import create_transform
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if loss_fn == "l2":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function does not exist")

    def preprocess(self, images, boxes, landmarks, return_faces=False):
        face_ls = []
        for bbox, lm in zip(boxes, landmarks):
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float,)
            face3d = self._face3d.preprocess(bbox, lm)
            face_ls.append(face3d)

        batch_inputs = []
        for image, face in zip(images, face_ls):
            face = self._face3d.forward(image, face)
            input_image = self._transform(face.normalized_image)
            batch_inputs.append(input_image)

        batch_inputs = torch.stack(batch_inputs, dim=0)
        if return_faces:
            return batch_inputs, face_ls
        else:
            return batch_inputs

    def postprocess(self, prediction, face):
        pass

    def forward(self, images, targets):

        preds = self._gaze_estimation_model.forward(images)
        # Regression loss
        loss = self.loss_fn(preds, targets)
        return loss

    def detect(self, x):
        with torch.no_grad():
            preds = self._gaze_estimation_model.forward(x)
        return preds.cpu().detach().numpy()

    def make_targets(self, predictions):
        return torch.from_numpy(predictions).to(self.device)

    def get_gaze_vector(self, predictions, faces):
        centers = []
        vectors = []
        for pred, face in zip(predictions, faces):
            face = self._face3d.postprocess([pred], face)
            centers.append(face.center)
            vectors.append(face.gaze_vector)
        return centers, vectors

    def _load_model(self) -> torch.nn.Module:
        model = create_model(self._config)
        checkpoint = torch.load(
            self._config.gaze_estimator.checkpoint, map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
        model.to(torch.device(self._config.device))
        model.eval()
        return model

