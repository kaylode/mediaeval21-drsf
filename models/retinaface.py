import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from .face_det.retinaface.models.retinaface import retinaface_mnet
from ..models.face_det.retinaface.layers.modules.multibox_loss import MultiBoxLoss
from ..models.face_det.retinaface.layers.functions.prior_box import PriorBox

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


class RetinaFaceDetector(nn.Module):
    def __init__(self):
        super(RetinaFaceDetector, self).__init__()

        self.model = retinaface_mnet(pretrained=True)
        self.config = self.model.cfg
    
    def preprocess(self, cv2_image):
        pil_image = Image.fromarray(cv2_image)
        np_image = np.uint8(pil_image)
        return np_image

    def forward(self, imgs, target_bboxes):
        loss_fn = DetectionLoss(self.config, image_size = imgs.shape[-2:])
        predictions = self.model.forward(imgs)
       
        # Multibox loss
        loss = loss_fn(predictions, target_bboxes) 
        return loss

    def detect(self, x):
        x_tensor = x.copy()
        results = self.model.detect(x_tensor) # xmin, ymin, xmax, ymax, scores
        return results