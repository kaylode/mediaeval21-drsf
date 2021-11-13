"""
"""

from omegaconf.dictconfig import DictConfig

from models.gaze_det.base import BaseModel

from models.gaze_det.ptgaze.common import Camera, Face
from models.gaze_det.ptgaze.head_pose_estimation import (
    HeadPoseNormalizer
)
from models.gaze_det.ptgaze.common.face_model_68 import FaceModel68


class Face3DModel(BaseModel):
    def __init__(self, config: DictConfig):
        super(Face3DModel, self).__init__()
        self._config = config
        self._face_model3d = FaceModel68()
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

