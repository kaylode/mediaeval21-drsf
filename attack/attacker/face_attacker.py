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

    def _generate_tensors(self, query):
        if not isinstance(query, list):
            query = [query]

        if isinstance(query[0], torch.Tensor):
            torch_images = query
        else:
            torch_images = [TFF.to_tensor(i) for i in query]

        return torch.stack(torch_images, dim=0).contiguous()

    def _generate_adv(self, images, face_boxes, deid_fn):
        """
        Generate deid image
        :params:
            images: list of cv2 image
            face_boxes: bounding boxes of face in the image. In (x1,y1,x2,y2) format
            deid_fn: De-identification method
        :return: deid cv2 image
        """
        deid = deid_fn.forward_batch(images, face_boxes)
        return deid

    def _generate_targets(self, victim, images):
        """
        Generate target for image using victim model
        :params:
            images: list of cv2 image
            victim: victim detection model
        :return: 
            face_box: bounding box of face in the image. In (x1,y1,x2,y2) format
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

        return face_boxes, targets

    def attack(self, victim, images, deid_fn, face_boxes=None, targets=None, optim_params={}):
        """
        Performs attack flow on image
        :params:
            images: list of cv2 images
            victim: victim detection model
            deid_fn: De-identification method
            face_boxes: boxes of faces
            targets: targets for image
            optim_params: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 image
        """
        # Generate target
        if face_boxes is None and targets is None:
            face_boxes, targets = self._generate_targets(victim, images)
        
        # De-id image with face box
        deid = self._generate_adv(images, face_boxes, deid_fn)
        deid_norm = victim.preprocess(deid) 

        # To tensor, allow gradients to be saved
        # if not isinstance(deid_norm, torch.Tensor):
        #     deid_tensor = TFF.to_tensor(deid_norm).contiguous()
        # else:
        #     deid_tensor = deid_norm.clone()   
        deid_tensor = self._generate_tensors(deid_norm)
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **optim_params)

        # Adversarial attack
        deid_tensor.requires_grad = True
        adv_res = self._iterative_attack(deid_tensor, targets, victim, optim, self.n_iter)

        # Postprocess, return cv2 image
        adv_res = victim.postprocess(adv_res)
        return adv_res