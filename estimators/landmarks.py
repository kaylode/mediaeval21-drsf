from typing import List

import torch
from models import face_align, face_det
from models.gaze_det.ptgaze.common import Face
from torchvision.transforms import functional as TFF


class LandmarkEstimator:
    def __init__(self, det_model, align_model):
        self.detector = det_model
        self.predictor = align_model

    @classmethod
    def from_name(cls, det_name: str, align_name: str) -> "LandmarkEstimator":
        det_model = face_det.get_model(det_name)
        align_model = face_align.get_model(align_name)
        return cls(det_model, align_model)

    def detect_faces(self, images: List) -> List[Face]:

        #  Get detection results
        results_norm = self.detector.preprocess(images)
        results_norm = self._generate_tensors(results_norm)
        det_results = self.detector.detect(results_norm)
        face_boxes = self.detector.get_face_boxes(det_results)

        ## WTF is this part mean?
        masks = [0 if len(box) == 0 else 1 for box in face_boxes]

        if sum(masks) == 0:
            # raise ValueError("Empty face bboxes")
            return [], []

        masked_face_boxes = [box for box, mask in zip(face_boxes, masks) if mask == 1]
        masked_images = [image for image, mask in zip(images, masks) if mask == 1]
        ## WTF is this part mean?

        # Detect on raw image
        centers, scales = self.predictor._get_scales_and_centers(masked_face_boxes)
        lm_norm = self.predictor.preprocess(masked_images, centers, scales)
        lm_norm = self._generate_tensors(lm_norm)
        _, landmarks = self.predictor.detect(lm_norm, centers, scales)

        # Mask empty prediction
        landmarks = [
            lm.numpy() if mask == 1 else np.zeros((68, 2))
            for lm, mask in zip(landmarks, masks)
        ]
        face_boxes = [
            box if mask == 1 else [0, 0, 0, 0] for box, mask in zip(face_boxes, masks)
        ]

        return face_boxes, landmarks

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
            torch_images = [
                TFF.to_tensor(i) if query.shape[-1] == 3 else torch.from_numpy(i)
                for i in query
            ]

        return torch.stack(torch_images, dim=0).contiguous()
