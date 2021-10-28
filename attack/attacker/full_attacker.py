import torch

from attack.algorithms import get_optim
from .base import Attacker

from models.face_align.models.face_alignment.utils import crop_tensor

class FullAttacker(Attacker):
    """
    Face model Attacker class
    :params:
        optim: name of attack algorithm
        n_iter: number of iterations
        eps: epsilon param
    """
    def __init__(self, optim, n_iter=10, eps=8/255.):
        super().__init__(optim, n_iter, eps)

    def _generate_targets(self, victim, images):
        """
        Generate target for image using victim model
        :params:
            images: list of cv2 image
            victim: victim detection model
        :return: 
            face_boxes: bounding boxes of face in the image. In (x1,y1,x2,y2) format
            targets: targets for image
        """

        # Normalize image
        query = victim.preprocess(images)

        # To tensor, allow gradients to be saved
        query_tensor = self._generate_tensors(query)

        # Detect on raw image
        predictions = victim.detect(query_tensor)

        # Make targets and face_box
        targets = victim.make_targets(predictions, images)
        face_boxes = victim.get_face_boxes(predictions)

        return targets, face_boxes

    def _generate_targets2(self, victim, images, centers, scales):
        """
        Generate target for image using victim model
        :params:
            images: cv2 image
            victim: victim detection model
            centers: center of face boxes
            scales: scales of face boxes
        :return: 
            targets: targets for images
        """

        # Normalize image
        query = victim.preprocess(images, centers, scales)
        query = self._generate_tensors(query)

        # Detect on raw image
        predictions = victim.detect(query, centers, scales)

        # Make targets and face_box
        targets = victim.make_targets(predictions)
  
        return targets

    def attack(self, victims, images, deid_images, targets=None, optim_params={}):
        """
        Performs attack flow on image
        :params:
            images: list of cv2 images
            victim: victim detection model
            deid_images: list of De-identification cv2 images
            targets: targets for image
            optim_params: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 images
        """
        # Generate target
        if targets is None:
            targets, face_boxes = self._generate_targets(victims[0], images)
        
        # Get centers and scales for preprocess
        centers, scales = victims[1]._get_scales_and_centers(face_boxes)
        lm_targets = self._generate_targets2(victims[1], images, centers, scales)

        deid_norm = victims[0].preprocess(deid_images) 

        deid_tensor = self._generate_tensors(deid_norm)
        deid_tensor.requires_grad = True
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **optim_params)

        for _ in range(self.n_iter):
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                det_loss = victims[0](deid_tensor, targets)
                
                lm_inputs = []
                for deid, center, scale in zip(deid_tensor, centers, scales):
                    lm_input = crop_tensor(deid, center, scale)
                    lm_inputs.append(lm_input)
                lm_inputs = torch.stack(lm_inputs, dim=0)

                lm_loss = victims[1](lm_inputs, lm_targets)
                
                if det_loss.item() > 1.0:
                    loss = lm_loss + det_loss
                else:
                    loss = lm_loss

                loss.backward()

            # if mask is not None:
            #     att_img.grad[mask] = 0

            optim.step()

        # Adversarial attack
        adv_res = deid_tensor.clone()

        # Postprocess, return cv2 image
        adv_res = victims[0].postprocess(adv_res)
        return adv_res