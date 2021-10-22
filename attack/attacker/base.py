import torch
from torchvision.transforms import functional as TFF

from attack.algorithms import get_optim

class Attacker:
    """
    Abstract class for Attacker
    :params:
        optim: name of attack algorithm
        n_iter: number of iterations
        eps: epsilon param
    """
    def __init__(self, optim, n_iter=10, eps=8/255.):
        self.n_iter = n_iter
        self.eps = eps
        self.optim = optim
    
    def _generate_adv(self, query_image):
        """
        Generate adversarial image
        :params:
            query_image: cv2 image
        :return: adversarial cv2 image
        """
        raise NotImplementedError("This is an interface method")

    def _generate_targets(self, victim, query_image):
        """
        Generate target for image using victim model
        :params:
            query_image: cv2 image
            victim: victim detection model
        :return: 
            targets: targets for image
        """
        raise NotImplementedError("This is an interface method")

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

            if mask is not None:
                att_img.grad[mask] = 0

            optim.step()

        results = att_img.clone()
        return results

    def attack(self, victim, query_image, targets=None, optim_params={}):
        """
        Performs attack flow on image
        :params:
            query_image: raw cv2 image
            victim: victim detection model
            targets: targets for image
            kwargs: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 image
        """
        # Generate target
        if targets is None:
            targets = self._generate_targets(victim, query_image)
        
        # Generate adverasarial
        adv_img = self._generate_adv(query_image)
        adv_norm = victim.preprocess(adv_img) 

        # To tensor, allow gradients to be saved
        adv_tensor = TFF.to_tensor(adv_norm).contiguous()
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[adv_tensor], epsilon=self.eps, **optim_params)

        # Adversarial attack
        adv_tensor.requires_grad = True
        adv_res = self._iterative_attack(adv_tensor, targets, victim, optim, self.n_iter)

        # Postprocess, return cv2 image
        adv_res = victim.postprocess(adv_res)
        return adv_res