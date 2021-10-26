"""
model from https://github.com/biubug6/Pytorch_Retinaface
"""

import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from .base import BaseDetector
from .models.retinaface.models.retinaface import retinaface_mnet
from .models.retinaface.layers.modules.multibox_loss import MultiBoxLoss
from .models.retinaface.layers.functions.prior_box import PriorBox

class DetectionLoss(nn.Module):
    def __init__(self, cfg, image_size):
        super(DetectionLoss, self).__init__()
        self.cfg = cfg
        self.multiboxloss = MultiBoxLoss(2, 0.45, True, 0, True, 7, 0.35, False)
        self.priorbox = PriorBox(cfg, image_size)
        with torch.no_grad():
            self.priorbox = self.priorbox.forward()

    def forward(self, predictions, targets):
        self.priorbox = self.priorbox.to(predictions[0].device)
        loss_l, loss_c, _ = self.multiboxloss(predictions, self.priorbox, targets)
        return  self.cfg['loc_weight'] * loss_l + loss_c


class RetinaFaceDetector(BaseDetector):
    def __init__(self):
        super(RetinaFaceDetector, self).__init__()

        self.model = retinaface_mnet(pretrained=True)
        self.model.eval()
        self.config = self.model.cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def preprocess(self, images):
        return images

    def postprocess(self, images):
        images = images.detach().numpy().squeeze().transpose((0,2,3,1))
        images = (images*255).astype(np.uint8)
        return images

    def forward(self, imgs, target_bboxes):
        loss_fn = DetectionLoss(self.config, image_size = imgs.shape[-2:])
        
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
        imgs = imgs.to(self.device)
        predictions = self.model.forward(imgs)
        # Multibox loss
        loss = loss_fn(predictions, target_bboxes) 
        return loss

    def detect(self, x):
        x_tensor = x.clone()
        if len(x_tensor.shape) == 3:
            x_tensor = x_tensor.unsqueeze(0)
        x_tensor = x_tensor.to(self.device)

        with torch.no_grad():
            results = self.model.detect(x_tensor) # xmin, ymin, xmax, ymax, scores
        return results

    def make_targets(self, predictions, images):

        batch_bboxes, batch_landmarks = predictions
        targets = []

        for box, landmark, image in zip(batch_bboxes, batch_landmarks, images):
            width, height = image.shape[1], image.shape[0]
            if box.shape[-1] != 5:
                box = torch.Tensor([[0, 0, width, height, 1]]).to(box.device)
                landmark = torch.zeros(1, 10)
            _target = torch.cat((box[:, :-1], landmark, box[:, -1:]), dim=-1)
            _target = _target.float().to(self.device)
            _target[:, -1] = 1
            _target[:, (0, 2)] /= width
            _target[:, (1, 3)] /= height

            targets.append(_target)

        return targets

    def get_face_boxes(self, predictions, return_probs=False):
        batch_bboxes, _ = predictions
        face_boxes = []
        for bboxes in batch_bboxes:
            face_box = bboxes.squeeze(0).numpy().astype(np.int).tolist()
            if not return_probs:
                face_box = face_box[:-1]
            face_boxes.append(face_box)
        return face_boxes