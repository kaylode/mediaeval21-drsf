import numpy as np
from .algorithms import get_optim

import torch
from torchvision.transforms import functional as TFF

class Attacker:
    def __init__(self, optim, n_iter=10, eps=8/255., eps_iter=2/255., alpha =0.05):
        self.n_iter = n_iter
        self.eps = eps
        self.eps_iter = eps_iter
        self.alpha = alpha
        self.optim = optim

    def _generate_deid(self, cv2_image, face_box, deid_fn):
        deid = deid_fn(cv2_image, face_box)
        return deid

    def _generate_targets(self, victim, cv2_image):
        # Normalize image
        query = victim.preprocess(cv2_image)

        # To tensor, allow gradients to be saved
        query_tensor = TFF.to_tensor(query).contiguous()

        # Detect on raw image
        predictions = victim.detect(query_tensor)

        # Make targets and face_box
        targets = victim.make_targets(predictions, cv2_image.shape[1], cv2_image.shape[0])
        face_box = victim.get_face_box(predictions)

        return face_box, targets

    def _iterative_attack(self, att_img, targets, model, optim, n_iter):

        for _ in range(n_iter):
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                loss = model(att_img, targets)
            loss.backward()
            # att_img.grad[mask] = 0
            optim.step()

        results = att_img.clone()
        return results

    def attack(self, victim, cv2_image, deid_fn, **kwargs):
        
        # Generate target
        face_box, targets = self._generate_targets(victim, cv2_image)
        
        # De-id image with face box
        deid = self._generate_deid(cv2_image, face_box, deid_fn)
        deid_norm = victim.preprocess(deid) 

        # To tensor, allow gradients to be saved
        deid_tensor = TFF.to_tensor(deid_norm).contiguous()
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **kwargs)

        deid_tensor.requires_grad = True
        adv_res = self._iterative_attack(deid_tensor, targets, victim, optim, self.n_iter)
        return adv_res