import torch
import torch.nn as nn

import numpy as np
from PIL import Image

from .face_det.retinaface.models.retinaface import retinaface_mnet
from .face_det.retinaface.layers.modules.multibox_loss import MultiBoxLoss
from .face_det.retinaface.layers.functions.prior_box import PriorBox

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
        
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
        predictions = self.model.forward(imgs)

        # Multibox loss
        loss = loss_fn(predictions, target_bboxes) 
        return loss

    def detect(self, x):
        x_tensor = x.clone()
        if len(x_tensor.shape) == 3:
            x_tensor = x_tensor.unsqueeze(0)
        results = self.model.detect(x_tensor) # xmin, ymin, xmax, ymax, scores
        return results

    def make_targets(self, predictions, width, height):
        bboxes, landmarks = predictions
        target = []
        for box, landmark in zip(bboxes, landmarks):
            if box.shape[-1] != 5:
                box = torch.Tensor([[0, 0, width, height, 1]]).to(box.device)
                landmark = torch.zeros(1, 10)
                
            _target = torch.cat((box[:, :-1], landmark, box[:, -1:]), dim=-1)
            _target = _target.float().to(box.device)
            _target[:, -1] = 1
            _target[:, (0, 2)] /= width
            _target[:, (1, 3)] /= height

            target.append(_target)

        return target

    def get_face_box(self, predictions):
        bboxes, _ = predictions
        face_box = bboxes[0].squeeze(0)[:-1].numpy().astype(np.int).tolist()
        return face_box