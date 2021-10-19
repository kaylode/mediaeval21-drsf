import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from .facenet import MTCNN

def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor

class MTCNNDetector(nn.Module):
    def __init__(self, image_size=160, thresholds=[0.6, 0.7, 0.7], loss_fn='l2'):
        super(MTCNNDetector, self).__init__()
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.optimizer = None

        if loss_fn == 'l2':
            self.loss_fn = nn.MSELoss()
        elif loss_fn == 'l1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError('Loss function does not exist')

        self.mtcnn = MTCNN(
            image_size=image_size, margin=0, min_face_size=20, select_largest =True,
            thresholds=thresholds, factor=0.709, post_process=True,
            device=device, keep_all=False
        )

    def preprocess(self, cv2_image):
        pil_image = Image.fromarray(cv2_image)
        np_image = np.uint8(pil_image)
        normalized = fixed_image_standardization(np_image)
        return normalized

    def forward(self, imgs, target_bboxes):
        n = target_bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')
        
        adv_det, adv_points = self.mtcnn.forward(imgs)
        # faces = model.extract(image, boxes, save_path=None)
        adv_scores = adv_det[:, -1]
        adv_bboxes = adv_det[:, :-1]

        num_preds = adv_bboxes.shape[0]
        target_bboxes = target_bboxes.repeat(num_preds, 1)
        
        # Regression loss
        loss = self.loss_fn(adv_bboxes, target_bboxes)
        
        return loss

    def detect(self, x):

        x_tensor = x.copy()
        _bboxes, _points = self.mtcnn.detect(x_tensor) # xmin, ymin, xmax, ymax, scores
        return _bboxes

    def compute_gradient(self, x, target_detections):

        x_tensor = torch.from_numpy(x)[None].cuda().float()
        x_tensor.requires_grad = True

        _bboxes = torch.from_numpy(target_detections[:, :-1]).float().cuda()
        _score = torch.from_numpy(target_detections[:, -1]).float()

        loss = self.forward(x_tensor, _bboxes)

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        self.mtcnn.zero_grad()
        loss.backward()
        return x_tensor.grad.data.cpu().numpy().squeeze(0)