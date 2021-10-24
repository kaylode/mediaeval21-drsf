"""
model from https://github.com/timesler/facenet-pytorch
"""

import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from .base import BaseAlignment
from .models.face_alignment import FaceAlignment, LandmarksType


class FANAlignment(BaseAlignment):
    def __init__(self, loss_fn='l2'):
        super(FANAlignment, self).__init__()
        
        if loss_fn == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError('Loss function does not exist')

        self.model = FaceAlignment(
            LandmarksType._2D,
            face_detector='dlib',
            flip_input=False,
            device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
        )

    def preprocess(self, cv2_image):
        # RGB Image
        return cv2_image

    def postprocess(self, image):
        return image

    def forward(self, imgs, target_bboxes):
        n = target_bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        adv_det, adv_points = self.model.forward(imgs)
        # faces = model.extract(image, boxes, save_path=None)
        adv_scores = adv_det[:, -1]
        adv_bboxes = adv_det[:, :-1]

        num_preds = adv_bboxes.shape[0]
        target_bboxes = target_bboxes.repeat(num_preds, 1)

        # Regression loss
        loss = self.loss_fn(adv_bboxes, target_bboxes)
        return loss

    def detect(self, x, face_box):
        x_tensor = x.clone()
        if len(x_tensor.shape) == 3:
            x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            _landmarks = self.model.get_landmarks(x_tensor, detected_faces=[face_box]) # 68 points
        return _landmarks

    def make_targets(self, predictions, image):
        return torch.from_numpy(predictions).cuda()