import torch
from torchvision.transforms import functional as TFF

from attack.algorithms import get_optim


def generate_tensors(query):
    """
    Generate tensors to allow computing gradients
    :params:
        query: list of cv2 image
    :return: torch tensors of images
    """
    if len(query.shape)==3:
        query = [query]

    if isinstance(query[0], torch.Tensor):
        torch_images = query
    else:
        torch_images = [
            TFF.to_tensor(i) if query.shape[-1] == 3 else torch.from_numpy(i) for i in query]

    return torch.stack(torch_images, dim=0).contiguous()

class Attacker:
    """
    Abstract class for Attacker
    :params:
        optim: name of attack algorithm
        max_iter: maximum number of iterations
        eps: epsilon param
    """
    def __init__(self, optim, max_iter=10, eps=8/255.):
        self.max_iter = max_iter
        self.eps = eps
        self.optim = optim
    
    def _generate_tensors(self, query):
        """
        Generate tensors to allow computing gradients
        :params:
            query: list of cv2 image
        :return: torch tensors of images
        """
        return generate_tensors(query)

    def _generate_adv(self, images):
        """
        Generate adversarial image
        :params:
            images: list of cv2 image
        :return: adversarial cv2 image
        """
        raise NotImplementedError("This is an interface method")

    def _generate_targets(self, victim, images):
        """
        Generate target for image using victim model
        :params:
            images: list of cv2 image
            victim: victim detection model
        :return: 
            targets: targets for image
        """
        raise NotImplementedError("This is an interface method")

    def _iterative_attack(self, att_img, targets, model, optim, max_iter, mask=None):
        """
        Performs iterative adversarial attack on image
        :params:
            att_img: input attack image
            targets: attack targets
            model: victim detection model
            optim: optimizer
            max_iter: number of attack iterations
            mask: gradient mask
        :return: 
            results: tensor image with updated gradients
        """

        for _ in range(max_iter):
            optim.zero_grad()
            with torch.set_grad_enabled(True):
                loss = model(att_img, targets)
            loss.backward()

            if mask is not None:
                att_img.grad[mask] = 0

            optim.step()

        results = att_img.clone()
        return results

    def attack(self, victim, query_images, targets=None, optim_params={}):
        """
        Performs attack flow on image
        :params:
            query_images: raw cv2 image
            victim: victim detection model
            targets: targets for image
            optim_params: keyword arguments that will be passed to optim
        :return: 
            adv_res: adversarial cv2 image
        """
        # Generate target
        if targets is None:
            targets = self._generate_targets(victim, query_images)
        
        # Generate adverasarial
        adv_imgs = self._generate_adv(query_images)
        adv_norm = victim.preprocess(adv_imgs) 

        # To tensor, allow gradients to be saved
        adv_tensors = self._generate_tensors(adv_norm)
        
        # Get attack algorithm
        optim = get_optim(self.optim, params=[adv_tensors], epsilon=self.eps, **optim_params)

        # Adversarial attack
        adv_tensors.requires_grad = True
        adv_res = self._iterative_attack(adv_tensors, targets, victim, optim, self.max_iter)

        # Postprocess, return cv2 image
        adv_res = victim.postprocess(adv_res)
        return adv_res