import torch
from torchvision.transforms import functional as TFF

from attack.algorithms import get_optim
from .base import Attacker

class FaceAttacker(Attacker):
    """
    Face model Attacker class
    :params:
        optim: name of attack algorithm
        n_iter: number of iterations
        eps: epsilon param
    """
    def __init__(self, optim, n_iter=10, eps=8/255.):
        super().__init__(optim, n_iter, eps)

    def _generate_adv(self, cv2_image, face_box, deid_fn):
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

    def attack(self, victim, cv2_image, deid_fn, face_box=None, targets=None, optim_params={}):
        """
        Performs attack flow on image
        :params:
            cv2_image: raw cv2 image
            victim: victim detection model
            deid_fn: De-identification method
            face_box: optimizer
            targets: targets for image
            optim_params: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 image
        """
        # Generate target
        if face_box is None and targets is None:
            face_box, targets = self._generate_targets(victim, cv2_image)
        
        # De-id image with face box
        deid = self._generate_adv(cv2_image, face_box, deid_fn)
        deid_norm = victim.preprocess(deid) 

        # To tensor, allow gradients to be saved
        if not isinstance(deid_norm, torch.Tensor):
            deid_tensor = TFF.to_tensor(deid_norm).contiguous()
        else:
            deid_tensor = deid_norm.clone()   
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **optim_params)

        # Adversarial attack
        deid_tensor.requires_grad = True
        adv_res = self._iterative_attack(deid_tensor, targets, victim, optim, self.n_iter)

        # Postprocess, return cv2 image
        adv_res = victim.postprocess(adv_res)
        return adv_res