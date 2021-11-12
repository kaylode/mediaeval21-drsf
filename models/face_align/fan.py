"""
model from https://github.com/1adrianb/face-alignment
"""

import torch
import torch.nn as nn

import numpy as np

from .base import BaseAlignment
from .models.face_alignment import (
    FaceAlignment,
    LandmarksType,
    crop,
    crop_mapping,
    get_preds_fromhm,
)


class FANAlignment(BaseAlignment):
    def __init__(self, loss_fn="l2"):
        super(FANAlignment, self).__init__()

        if loss_fn == "l2":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function does not exist")

        self.model = FaceAlignment(
            LandmarksType._2D,
            flip_input=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_scales_and_centers(self, face_boxes):
        centers = []
        scales = []
        for face_box in face_boxes:
            center = torch.tensor(
                [
                    face_box[2] - (face_box[2] - face_box[0]) / 2.0,
                    face_box[3] - (face_box[3] - face_box[1]) / 2.0,
                ]
            )
            center[1] = center[1] - (face_box[3] - face_box[1]) * 0.12
            scale = (
                face_box[2] - face_box[0] + face_box[3] - face_box[1]
            ) / 195  # 195 is dlib reference scale
            centers.append(center)
            scales.append(scale)

        return centers, scales

    def preprocess(self, images, centers, scales, return_points=False):
        # RGB Image
        batch_inputs = []
        new_boxes = []
        old_boxes = []
        old_shapes = []
        for image, center, scale in zip(images, centers, scales):
            cropped, old_box, new_box, old_shape = crop(
                image, center, scale, return_points=True
            )
            inputs = torch.from_numpy(cropped.transpose((2, 0, 1))).float()
            inputs.div_(255.0)
            batch_inputs.append(inputs)
            new_boxes.append(new_box)
            old_boxes.append(old_box)
            old_shapes.append(old_shape)
        batch_inputs = np.stack(batch_inputs, axis=0)

        if return_points:
            return batch_inputs, new_boxes, old_boxes, old_shapes
        else:
            return batch_inputs

    def postprocess(self, ori_images, crop_images, old_boxes, new_boxes, old_shapes):

        crop_images = crop_images.detach().numpy().transpose((0, 2, 3, 1))
        images = []
        for ori_image, crop_image, old_box, new_box, old_shape in zip(
            ori_images, crop_images, old_boxes, new_boxes, old_shapes
        ):
            old_width, old_height = old_shape
            crop_image = (crop_image * 255).astype(np.uint8)
            cv2_image = crop_mapping(
                ori_image, crop_image, old_box, new_box, old_width, old_height
            )
            images.append(cv2_image)

        return images

    def forward(self, imgs, targets):
        preds = self.model.forward(imgs)
        # Regression loss
        loss = self.loss_fn(preds, targets)
        return loss

    def detect(self, x, centers, scales):
        with torch.no_grad():
            heatmaps = self.model.forward(x)
        heatmaps = heatmaps.detach().cpu().numpy()
        batch_landmarks = []
        for hm, center, scale in zip(heatmaps, centers, scales):
            hm = np.expand_dims(hm, axis=0)
            _, preds, _ = get_preds_fromhm(hm, center.numpy(), scale)
            landmarks = torch.from_numpy(preds).view(68, 2)  # 68 keypoints
            batch_landmarks.append(landmarks)
        return heatmaps, batch_landmarks

    def make_targets(self, predictions):
        return torch.from_numpy(predictions[0]).to(self.device)

    def get_landmarks(self, predictions):
        return predictions[1]
