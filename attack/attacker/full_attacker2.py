import torch

from attack.algorithms import get_optim
from .base import Attacker

from models.face_align.models.face_alignment.utils import crop_tensor


class FullAttacker2(Attacker):
    """
    Face model Attacker class
    :params:
        optim: name of attack algorithm
        n_iter: number of iterations
        eps: epsilon param
    """

    def __init__(self, optim, n_iter=10, eps=8 / 255.0):
        super().__init__(optim, n_iter, eps)

    def _generate_targets(self, victims, images):
        """
        Generate target for image using victim models
        :params:
            images: list of cv2 image
            victims: dictionary of victim models.  
        :return: 
            face_boxes: bounding boxes of face in the image. In (x1,y1,x2,y2) format
            targets: targets for image
        """

        # Generate detection targets
        # Normalize image
        query = victims["detection"].preprocess(images)

        # To tensor, allow gradients to be saved
        query_tensor = self._generate_tensors(query)

        # Detect on raw image
        predictions = victims["detection"].detect(query_tensor)

        # Make targets and face_box
        det_targets = victims["detection"].make_targets(predictions, images)
        face_boxes = victims["detection"].get_face_boxes(predictions)

        # Check if a box is empty, if so, use previous box or next box
        for idx, box in enumerate(face_boxes):
            if len(box) == 0:
                face_boxes[idx] = face_boxes[idx - 1][:]

        # Generate alignment targets
        # Get scales and centers of face boxes
        centers, scales = victims["alignment"]._get_scales_and_centers(face_boxes)

        # Normalize image
        query = victims["alignment"].preprocess(images, centers, scales)
        query = self._generate_tensors(query)

        # Detect on raw image
        predictions = victims["alignment"].detect(query, centers, scales)

        # Make targets
        lm_targets = victims["alignment"].make_targets(predictions)
        landmarks = victims["alignment"].get_landmarks(predictions)

        landmarks = [lm.numpy() for lm in landmarks]
        # Generate gaze targets
        query = victims["gaze"].preprocess(images, face_boxes, landmarks)

        predictions = victims["gaze"].detect(query)
        gaze_targets = victims["gaze"].make_targets(predictions)

        return {
            "detection": det_targets,
            "alignment": lm_targets,
            "gaze": gaze_targets,
            "gaze_boxes": face_boxes,
            "gaze_landmarks": landmarks,
            "alignment_centers": centers,
            "alignment_scales": scales,
        }

    def _iterative_attack(self, att_imgs, targets, victims, optim, n_iter, mask=None):
        """
        Performs iterative adversarial attack on batch images
        :params:
            att_imgs: input attack image
            targets: dictionary of attack targets
            victims: dictionary of victim models.  
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

                # Forward face detection model
                det_loss = victims["detection"](att_imgs, targets["detection"])

                # Generate cropped tensors to prepare for alignment model
                lm_inputs = []
                for deid, center, scale in zip(
                    att_imgs, targets["alignment_centers"], targets["alignment_scales"]
                ):
                    lm_input = crop_tensor(deid, center, scale)
                    lm_inputs.append(lm_input)
                lm_inputs = torch.stack(lm_inputs, dim=0)

                # Forward alignment model
                lm_loss = victims["alignment"](lm_inputs, targets["alignment"])
                
                # Generate tensors for gaze model
                gaze_inputs = victims["gaze"].preprocess(att_imgs, targets["gaze_boxes"], targets["gaze_landmarks"])
                gaze_loss = victims["gaze"](gaze_inputs, targets["gaze"])

                # Sum up loss
                loss = lm_loss*0.35 + det_loss*0.55 + gaze_loss*0.2
                loss.backward()

            if mask is not None:
                att_imgs.grad[mask] = 0

            optim.step()

        # Get the adversarial images
        adv_res = att_imgs.clone()
        return adv_res

    def attack(self, victims, images, deid_images, optim_params={}):
        """
        Performs attack flow on image
        :params:
            images: list of rgb cv2 images
            victims: dictionary of victim models.  
            deid_images: list of De-identification cv2 images
            optim_params: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 images
        """

        # assert (
        #     "detection" in victims.keys() and "alignment" in victims.keys()
        # ), "Need both detection and alignment models to attack"

        targets = self._generate_targets(victims, images)

        # Process deid images for detection model
        deid_norm = victims["detection"].preprocess(deid_images)

        # To tensors and turn on gradients flow
        deid_tensor = self._generate_tensors(deid_norm)
        deid_tensor.requires_grad = True

        # Get attack algorithm
        optim = get_optim(
            self.optim, params=[deid_tensor], epsilon=self.eps, **optim_params
        )

        # Start iterative attack
        adv_res = self._iterative_attack(
            deid_tensor,
            targets=targets,
            victims=victims,
            optim=optim,
            n_iter=self.n_iter,
        )

        # Postprocess, return cv2 image
        adv_images = victims["detection"].postprocess(adv_res)
        return adv_images