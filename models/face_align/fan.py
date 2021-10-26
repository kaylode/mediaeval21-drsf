"""
model from https://github.com/timesler/facenet-pytorch
"""

import torch
import torch.nn as nn

import numpy as np

from .base import BaseAlignment
from .models.face_alignment import FaceAlignment, LandmarksType, crop, crop_mapping, get_preds_fromhm

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
            flip_input=False,
            device= 'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _get_scale_and_center(self, face_box):
        center = torch.tensor(
            [face_box[2] - (face_box[2] - face_box[0]) / 2.0, face_box[3] - (face_box[3] - face_box[1]) / 2.0])
        center[1] = center[1] - (face_box[3] - face_box[1]) * 0.12
        scale = (face_box[2] - face_box[0] + face_box[3] - face_box[1]) / 195 # 195 is dlib reference scale
        return center, scale

    def preprocess(self, cv2_image, center, scale):
        # RGB Image
        cropped, old_box, new_box, old_shape = crop(cv2_image, center, scale, return_points=True)
        inputs = torch.from_numpy(cropped.transpose((2, 0, 1))).float()
        inputs.div_(255.0).unsqueeze_(0)
        return inputs, new_box, old_box, old_shape

    def postprocess(self, ori_image, crop_image, old_box, new_box, old_shape):
        old_width, old_height = old_shape
        crop_image = crop_image.detach().numpy().squeeze().transpose((1,2,0))
        crop_image = (crop_image*255).astype(np.uint8)
        cv2_image = crop_mapping(ori_image, crop_image, old_box, new_box, old_width, old_height)
        return cv2_image

    def forward(self, imgs, targets):
        preds = self.model.forward(imgs)
        # Regression loss
        loss = self.loss_fn(preds, targets)
        return loss

    def detect(self, x, center, scale):
        with torch.no_grad():
            heatmaps = self.model.forward(x) 
        heatmaps = heatmaps.detach().cpu().numpy()
        _, preds, _ = get_preds_fromhm(heatmaps, center.numpy(), scale) 
        landmarks = torch.from_numpy(preds).view(68, 2) # 68 keypoints

        return heatmaps, landmarks

    def make_targets(self, predictions):
        return torch.from_numpy(predictions[0]).to(self.device)