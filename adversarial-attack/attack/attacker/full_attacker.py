import torch

from attack.algorithms import get_optim
from .base import Attacker

from models.face_align.models.face_alignment.utils import crop_tensor


class FullAttacker(Attacker):
    """
    Adversarial attack on Face Detection / Landmark Estimation / Gaze Estimation models
    :params:
        optim: name of attack algorithm
        max_iter: maximum number of iterations
        eps: epsilon param
    """

    def __init__(self, optim, max_iter=150, eps=8 / 255.0):
        super().__init__(optim, max_iter, eps)

    def _generate_targets(self, victims, images):
        """
        Generate target for image using victim models
        :params:
            images: list of cv2 image
            victims: dictionary of victim models.  
        :return: 
            targets_dict: dict of targets which required for _iterative_attack method
        """

        targets_dict = {}

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

        targets_dict["detection"] = det_targets

        # Generate alignment targets
        # Get scales and centers of face boxes
        if "alignment" in victims.keys():
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

            targets_dict["alignment"] = lm_targets
            targets_dict["alignment_centers"] = centers
            targets_dict["alignment_scales"] = scales

            # Generate gaze targets
            if "gaze" in victims.keys():
                query = victims["gaze"].preprocess(images, face_boxes, landmarks)
                predictions = victims["gaze"].detect(query)
                gaze_targets = victims["gaze"].make_targets(predictions)

                targets_dict["gaze"] = gaze_targets
                targets_dict["gaze_boxes"] = face_boxes
                targets_dict["gaze_landmarks"] = landmarks

        return targets_dict

    def _iterative_attack(self, att_imgs, targets, victims, optim, max_iter, mask=None):
        """
        Performs iterative adversarial attack on batch images
        :params:
            att_imgs: input attack image
            targets: dictionary of attack targets
            victims: dictionary of victim models.  
            optim: optimizer
            max_iter: maximum number of attack iterations
            mask: gradient mask
        :return: 
            results: tensor image with updated gradients
        """

        # Batch size for normalizing loss
        batch_size = att_imgs.shape[0]

        iter = 0
        # Start attack
        while True:
            optim.zero_grad()
            with torch.set_grad_enabled(True):

                # Forward face detection model
                det_loss = victims["detection"](att_imgs, targets["detection"])

                if "alignment" in victims.keys():
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
                    del lm_inputs

                    if "gaze" in victims.keys():
                        # Generate tensors for gaze model
                        gaze_inputs = victims["gaze"].preprocess(att_imgs, targets["gaze_boxes"], targets["gaze_landmarks"])
                        gaze_loss = victims["gaze"](gaze_inputs, targets["gaze"])
                        del gaze_inputs
                
                # Sum up loss
                if det_loss.item() / batch_size > 0.3:
                    loss = det_loss
                elif "alignment" in victims.keys() and lm_loss.item() / batch_size > 3e-5:
                    loss = lm_loss + det_loss
                elif "gaze" in victims.keys() and gaze_loss.item()/ batch_size > 5e-4:
                    loss = gaze_loss + lm_loss + det_loss
                else:
                    break

                loss.backward()

            if mask is not None:
                att_imgs.grad[mask] = 0

            optim.step()

            if iter == max_iter:
                break

            iter += 1
        
        print("Number of iter: ", iter)
        # Get the adversarial images
        att_imgs = att_imgs.detach().cpu()
        return att_imgs

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
            max_iter=self.max_iter,
        )

        # Postprocess, return cv2 image
        adv_images = victims["detection"].postprocess(adv_res)
        return adv_images
