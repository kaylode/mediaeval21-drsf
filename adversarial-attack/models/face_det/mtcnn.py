"""
model from https://github.com/timesler/facenet-pytorch
"""

import torch
import torch.nn as nn

import numpy as np

from .base import BaseDetector
from .models.facenet import MTCNN


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


class MTCNNDetector(BaseDetector):
    def __init__(self, image_size=160, thresholds=[0.6, 0.7, 0.7], loss_fn="l2"):
        super(MTCNNDetector, self).__init__()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if loss_fn == "l2":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "l1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError("Loss function does not exist")

        self.model = MTCNN(
            image_size=image_size,
            margin=0,
            min_face_size=20,
            select_largest=True,
            thresholds=thresholds,
            factor=0.709,
            post_process=True,
            device=device,
            keep_all=False,
        )

    def preprocess(self, images):
        
        images = np.stack(images, axis=0)
        return images

    def postprocess(self, images):
        images = images.detach().numpy()
        unnormalized = np.clip(images*255, 0, 255).astype(np.uint8)
        images = unnormalized.transpose((0, 2, 3, 1))
        return images

    def forward(self, imgs, target_feats):
        adv_det, adv_points, feats = self.model.forward(imgs, target_feats)

        # Regression loss
        loss = 0
        for stage_feat, stage_target_feat in zip(feats, target_feats):
            for feat, target_feat in zip(stage_feat, stage_target_feat):
                feat = feat['f']
                target_feat = target_feat['f']
                for f, t in zip(feat, target_feat):
                    loss += self.loss_fn(f, t)
        return loss

    def detect(self, x):
        x_tensor = x.clone()
        if len(x_tensor.shape) == 3:
            x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            _bboxes, _points, feats = self.model.detect(
                x_tensor
            )  # xmin, ymin, xmax, ymax, scores

        return _bboxes, feats

    def make_targets(self, predictions, images):
        return predictions[1]

    def get_face_boxes(self, predictions, return_probs=False):
        batch_bboxes = predictions[0]
        face_boxes = []
        for bboxes in batch_bboxes:
            if bboxes.shape[0] == 0:
                face_box = []
            else:
                face_box = bboxes.squeeze(0).astype(np.int).tolist()
            if not return_probs:
                face_box = face_box[:-1]
            face_boxes.append(face_box)
        return face_boxes
