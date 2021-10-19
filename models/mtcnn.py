import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from PIL import Image

from .base import BaseDetector
from .face_det.facenet import MTCNN

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class MTCNNDetector(BaseDetector):
    def __init__(self, image_size=160, thresholds=[0.6, 0.7, 0.7], loss_fn='l2'):
        super(MTCNNDetector, self).__init__()
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if loss_fn == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError('Loss function does not exist')

        self.model = MTCNN(
            image_size=image_size, margin=0, min_face_size=20, select_largest =True,
            thresholds=thresholds, factor=0.709, post_process=True,
            device=device, keep_all=False
        )

    def preprocess(self, cv2_image):
        pil_image = Image.fromarray(cv2_image)
        np_image = np.uint8(pil_image)
        normalized = fixed_image_standardization(np_image)
        return normalized

    def postprocess(self, image):
        image = image.detach().numpy()
        unnormalized = np.clip(image * 128.0 + 127.5, 0, 255).astype(np.uint8)
        cv2_image = unnormalized.squeeze().transpose((1,2,0))
        return cv2_image

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

    def detect(self, x):
        x_tensor = x.clone()
        if len(x_tensor.shape) == 3:
            x_tensor = x_tensor.unsqueeze(0)

        with torch.no_grad():
            _bboxes, _points = self.model.detect(x_tensor) # xmin, ymin, xmax, ymax, scores
        return _bboxes

    def make_targets(self, predictions, image):
        return torch.from_numpy(predictions[:, :-1]).cuda()

    def get_face_box(self, predictions, return_probs=False):
        bboxes = predictions
        face_box = bboxes.squeeze(0).astype(np.int).tolist()
        if not return_probs:
            face_box = face_box[:-1]
        return face_box