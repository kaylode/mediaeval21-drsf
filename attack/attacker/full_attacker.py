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
        Generate target for image using victim detection model
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
        Generate target for image using victim alignment model
        :params:
            images: cv2 image
            victim: victim alignment model
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

        # Make targets
        targets = victim.make_targets(predictions)
  
        return targets

    def _iterative_attack(self, att_imgs, targets, victims, optim, n_iter, mask=None):
        """
        Performs iterative adversarial attack on batch images
        :params:
            att_imgs: input attack image
            targets: attack targets
            victims: list of models. Must be in order of detection model then alignment model
            optim: optimizer
            n_iter: number of attack iterations
            mask: gradient mask
        :return: 
            results: tensor image with updated gradients
        """
        
        # Batch size for normalizing loss
        batch_size = att_imgs.shape[0]
        
        # Start attack
        for _ in range(n_iter):
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                det_loss = victims[0](att_imgs, targets[0])
                
                lm_inputs = []
                for deid, center, scale in zip(att_imgs, targets[2], targets[4]):
                    lm_input = crop_tensor(deid, center, scale)
                    lm_inputs.append(lm_input)
                lm_inputs = torch.stack(lm_inputs, dim=0)

                lm_loss = victims[1](lm_inputs, targets[1])
                
                if det_loss.item()/batch_size > 1.0:
                    loss = lm_loss + det_loss
                else:
                    loss = lm_loss

                loss.backward()

            if mask is not None:
                att_imgs.grad[mask] = 0

            optim.step()

        # Adversarial attack
        adv_res = att_imgs.clone()
        return adv_res
        

    def attack(self, victims, images, deid_images, optim_params={}):
        """
        Performs attack flow on image
        :params:
            images: list of rgb cv2 images
            victims: list of models. Must be in order of detection model then alignment model
            deid_images: list of De-identification cv2 images
            optim_params: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 images
        """
        # Generate detection targets
        det_targets, face_boxes = self._generate_targets(victims[0], images)
        
        # Generate alignment targets
        centers, scales = victims[1]._get_scales_and_centers(face_boxes)
        lm_targets = self._generate_targets2(victims[1], images, centers, scales)

        # Process deid images for detection model
        deid_norm = victims[0].preprocess(deid_images) 

        # To tensors and turn on gradients flow
        deid_tensor = self._generate_tensors(deid_norm)
        deid_tensor.requires_grad = True
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **optim_params)

        # Start iterative attack
        adv_res = self._iterative_attack(
            deid_tensor,
            targets = [det_targets, lm_targets, centers, scales],
            victims = victims,
            optim=optim,
            n_iter=self.n_iter)

        # Postprocess, return cv2 image
        adv_res = victims[0].postprocess(adv_res)
        return adv_res