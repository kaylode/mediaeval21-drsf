import numpy as np
from typing import Dict, List

import torch
from models import face_align, face_det, gaze_det
from models.gaze_det.ptgaze.common import Face
from torchvision.transforms import functional as TFF
from helper.camera import Camera
from helper.face_parts import FacePartsName
from helper.face import Face

from omegaconf.dictconfig import DictConfig


class GazeEstimator:
    def __init__(self, det_model, align_model, face3d_model, gaze_model):
        self.detector = det_model
        self.face3d = face3d_model
        self.lm_predictor = align_model
        self.gaze_predictor = gaze_model

        self.camera = self.face3d.camera

    @classmethod
    def from_name(
        cls,
        det_name: str,
        align_name: str,
        face3d_name: str,
        gaze_name: str,
        cfg: DictConfig,
    ) -> "GazeEstimator":
        det_model = face_det.get_model(det_name)
        align_model = face_align.get_model(align_name)
        test_params = {
            "mode": "ETH-XGaze",
            "device": "cpu",
            "model": {"name": "resnet18"},
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
        config = cfg

        face3d_model = gaze_det.get_model(face3d_name, config=config)
        gaze_model = gaze_det.get_model(gaze_name, config=config)
        return cls(det_model, align_model, face3d_model, gaze_model)

    @torch.no_grad()
    def detect_faces(self, images: List) -> List[Face]:
        # debug trace
        #  Get detection results
        results_norm = self.detector.preprocess(images)
        results_norm = self._generate_tensors(results_norm)
        det_results = self.detector.detect(results_norm)
        face_boxes = self.detector.get_face_boxes(det_results)

        ## WTF is this part mean?
        masks = [0 if len(box) == 0 else 1 for box in face_boxes]

        if sum(masks) == 0:
            # raise ValueError("Empty face bboxes")
            return []

        masked_face_boxes = [box for box, mask in zip(face_boxes, masks) if mask == 1]
        masked_images = [image for image, mask in zip(images, masks) if mask == 1]
        ## WTF is this part mean?

        # Detect on raw image

        centers, scales = self.lm_predictor._get_scales_and_centers(masked_face_boxes)
        lm_norm = self.lm_predictor.preprocess(masked_images, centers, scales)
        lm_norm = self._generate_tensors(lm_norm)
        _, landmarks = self.lm_predictor.detect(lm_norm, centers, scales)

        # Mask empty prediction
        landmarks = [
            lm.numpy() if mask == 1 else np.zeros((68, 2))
            for lm, mask in zip(landmarks, masks)
        ]
        face_boxes = [
            box if mask == 1 else [0, 0, 0, 0] for box, mask in zip(face_boxes, masks)
        ]
        face_ls = []
        for bbox, lm in zip(face_boxes, landmarks):
            bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float,)
            face3d = self.face3d.preprocess(bbox, lm)
            face_ls.append(face3d)
        return face_ls

    def estimate_single_gaze(self, image, face: Face):
        if isinstance(image, List):
            image = image[0]
        face = self.face3d.forward(image, face)
        data = self.gaze_predictor.preprocess(face)
        gaze_vector = self.gaze_predictor.detect(data)
        self.face3d.postprocess(gaze_vector, face)
        return gaze_vector

    def estimate_gaze(self, image, face):
        return self.estimate_single_gaze(image, face)

    @staticmethod
    def _generate_tensors(query):
        """
        Generate tensors to allow computing gradients
        :params:
            query: list of cv2 image
        :return: torch tensors of images
        """
        if len(query.shape) == 3:
            query = [query]

        if isinstance(query[0], torch.Tensor):
            torch_images = query
        else:
            # torch_images = [
            #     TFF.to_tensor(i) if query.shape[-1] == 3 else torch.from_numpy(i)
            #     for i in query
            # ]
            # wrong code
            if query[0].shape[-1] == 3:
                torch_images = [TFF.to_tensor(i).float() for i in query]
            else:
                torch_images = [torch.from_numpy(i).float() for i in query]
        return torch.stack(torch_images, dim=0).contiguous()
