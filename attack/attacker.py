from .algorithms import get_optim

import torch
from torchvision.transforms import functional as TFF

class Attacker:
    """
    Attacker class
    :params:
        optim: name of attack algorithm
        n_iter: number of iterations
        eps: epsilon param
    """
    def __init__(self, optim, n_iter=10, eps=8/255.):
        self.n_iter = n_iter
        self.eps = eps
        self.optim = optim

    def _generate_deid(self, cv2_image, face_box, deid_fn):
        """
        Generate deid image
        :params:
            cv2_image: cv2 image
            face_box: bounding box of face in the image. In (x1,y1,x2,y2) format
            deid_fn: De-identification method
        :return: deid cv2 image
        """
        deid = deid_fn(cv2_image, face_box)
        return deid

    def _generate_targets(self, victim, cv2_image):
        """
        Generate target for image using victim model
        :params:
            cv2_image: cv2 image
            victim: victim detection model
        :return: 
            face_box: bounding box of face in the image. In (x1,y1,x2,y2) format
            targets: targets for image
        """

        # Normalize image
        query = victim.preprocess(cv2_image)

        # To tensor, allow gradients to be saved
        query_tensor = TFF.to_tensor(query).contiguous()

        # Detect on raw image
        predictions = victim.detect(query_tensor)

        # Make targets and face_box
        targets = victim.make_targets(predictions, cv2_image)
        face_box = victim.get_face_box(predictions)

        return face_box, targets

    def _iterative_attack(self, att_img, targets, model, optim, n_iter, mask=None):
        """
        Performs iterative adversarial attack on image
        :params:
            att_img: input attack image
            targets: attack targets
            model: victim detection model
            optim: optimizer
            n_iter: number of attack iterations
            mask: gradient mask
        :return: 
            results: tensor image with updated gradients
        """

        for _ in range(n_iter):
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                loss = model(att_img, targets)
            loss.backward()
            # att_img.grad[mask] = 0
            optim.step()

        results = att_img.clone()
        return results

    def attack(self, victim, cv2_image, deid_fn, face_box=None, targets=None, **kwargs):
        """
        Performs attack flow on image
        :params:
            cv2_image: raw cv2 image
            victim: victim detection model
            deid_fn: De-identification method
            face_box: optimizer
            targets: targets for image
            kwargs: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 image
        """
        # Generate target
        if face_box is None and targets is None:
            face_box, targets = self._generate_targets(victim, cv2_image)
        
        # De-id image with face box
        deid = self._generate_deid(cv2_image, face_box, deid_fn)
        deid_norm = victim.preprocess(deid) 

        # To tensor, allow gradients to be saved
        deid_tensor = TFF.to_tensor(deid_norm).contiguous()
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **kwargs)

        # Adversarial attack
        deid_tensor.requires_grad = True
        adv_res = self._iterative_attack(deid_tensor, targets, victim, optim, self.n_iter)

        # Postprocess, return cv2 image
        adv_res = victim.postprocess(adv_res)
        return adv_res