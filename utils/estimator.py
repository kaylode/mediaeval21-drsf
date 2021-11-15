import numpy as np
from typing import List

import torch
from models import face_align, face_det, gaze_det
from models.gaze_det.ptgaze.common import Face
from torchvision.transforms import functional as TFF

class Estimator:
    def __init__(self, det_model, align_model, gaze_model):
        self.detector = det_model
        self.face3d = gaze_model._face3d
        self.lm_predictor = align_model
        self.gaze_predictor = gaze_model

        self.camera = self.face3d.camera

    @classmethod
    def from_name(
        cls,
        det_name: str,
        align_name: str,
        gaze_name: str,
        width: int, height: int
    ) -> "Estimator":
        det_model = face_det.get_model(det_name)
        align_model = face_align.get_model(align_name)

        gaze_model = gaze_det.get_model("GazeModel", name=gaze_name, width=width, height=height)
        return cls(det_model, align_model, gaze_model)

    @torch.no_grad()
    def detect_faces(self, images: List) -> List[Face]:
        results_norm = self.detector.preprocess(images)
        results_norm = self._generate_tensors(results_norm)
        det_results = self.detector.detect(results_norm)
        face_boxes = self.detector.get_face_boxes(det_results)

        masks = [0 if len(box) == 0 else 1 for box in face_boxes]

        if sum(masks) == 0:
            return []

        masked_face_boxes = [box for box, mask in zip(face_boxes, masks) if mask == 1]
        masked_images = [image for image, mask in zip(images, masks) if mask == 1]

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
        
        inputs, face_ls = self.gaze_predictor.preprocess(
                masked_images, 
                face_boxes, 
                landmarks, return_faces=True
        )

        return inputs, face_ls[0]

    def estimate_single_gaze(self, input, face):
        gaze_vectors = self.gaze_predictor.detect(input)
        self.face3d.postprocess(gaze_vectors, face)
        return gaze_vectors

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
            if query[0].shape[-1] == 3:
                torch_images = [TFF.to_tensor(i).float() for i in query]
            else:
                torch_images = [torch.from_numpy(i).float() for i in query]
        return torch.stack(torch_images, dim=0).contiguous()