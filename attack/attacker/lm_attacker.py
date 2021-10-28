from attack.algorithms import get_optim
from .base import Attacker

class LandmarkAttacker(Attacker):
    """
    Landmark model Attacker class
    :params:
        optim: name of attack algorithm
        n_iter: number of iterations
        eps: epsilon param
    """
    def __init__(self, optim, n_iter=10, eps=8/255.):
        super().__init__(optim, n_iter, eps)

    def _generate_targets(self, victim, images, centers, scales):
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

    def attack(self, victim, images, deid_images, face_boxes, targets=None, optim_params={}):
        """
        Performs attack flow on image
        :params:
            images: list of cv2 images
            victim: victim detection model
            deid_images: list of De-identification cv2 images
            face_boxes: boxes of faces
            targets: targets for images
            optim_params: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 images
        """

        # Get centers and scales for preprocess
        centers, scales = victim._get_scales_and_centers(face_boxes)

        # Generate targets
        if targets is None:
            targets = self._generate_targets(victim, images, centers, scales)
        
        # De-id images
        deid_norm, new_boxes, old_boxes, old_shapes = victim.preprocess(deid_images, centers, scales, return_points=True) 

        deid_tensor = self._generate_tensors(deid_norm)
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[deid_tensor], epsilon=self.eps, **optim_params)

        # Adversarial attack
        deid_tensor.requires_grad = True
        adv_res = self._iterative_attack(deid_tensor, targets, victim, optim, self.n_iter)

        # Postprocess, return adversarial images
        adv_res = victim.postprocess(
            deid_images, adv_res, 
            old_boxes, new_boxes, old_shapes)

        return adv_res