import numpy as np
from .deid import Pixelate, Blur
from .algorithms import get_optim

import torch
import torch.nn.functional as F

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
        query = victim.preprocess(cv2_image)
        targets = victim.detect(query)
        return targets

    def _iterative_attack(self, img, targets, model, optim, n_iter):
        att_img = img.clone()
        att_img.requires_grad = True

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
        targets = self._generate_targets(victim, cv2_image)

        # Generate queries
        face_box = targets.squeeze(0)[:-1].astype(np.int).tolist()
        deid = self._generate_deid(cv2_image, face_box, deid_fn)
        deid_norm = victim.preprocess(deid) # Preprocess deid image

        # To tensor, allow gradients to be saved
        deid_tensor = F.to_tensor(deid_norm).contiguous()
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **kwargs)

        adv_res = self._iterative_attack(deid_tensor, targets, victim, optim, self.n_iter)
        return adv_res