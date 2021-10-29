"""
Copied from https://github.com/honguyenhaituan/PrivacyPreservingFaceRecognition/blob/16afc9df67afafe626a42ae7a5173547e9adae21/attacks/optim.py#L80
author: honguyenhaituan
"""

import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
from torch.optim.optimizer import Optimizer

class I_FGSM: 
    def __init__(self, params, epsilon=8/255., alpha=1/255., min_value=0, max_value=1): 
        self.params = params
        self.epsilon = epsilon
        self.alpha = alpha
        self.min_value = min_value
        self.max_value = max_value
        self.updated_params = []
        for param in self.params:
            self.updated_params.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        return -self.alpha * torch.sign(self.params[idx].grad)

    @torch.no_grad()
    def step(self):
        for idx, (param, updated_param) in enumerate(zip(self.params, self.updated_params)):
            if param is None: 
                continue
    
            n_update = torch.clamp(updated_param + self._cal_update(idx), -self.epsilon, self.epsilon)
            update = n_update - updated_param
            n_param = torch.clamp(param + update, self.min_value, self.max_value)
            update = n_param - param

            param += update
            updated_param += update

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class MI_FGSM(I_FGSM):
    def __init__(self, params, epsilon=8/255., momemtum=0, **kwargs):
        super(MI_FGSM, self).__init__(params, epsilon, **kwargs)
        self.momentum = momemtum
        self.o_grad = []
        for param in self.params:
            self.o_grad.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        grad = self.o_grad[idx] * self.momentum + self.params[idx].grad / torch.sum(torch.abs(self.params[idx].grad))
        return -self.alpha * torch.sign(grad)

    def zero_grad(self):
        for o_grad, param in zip(self.o_grad, self.params):
            if param.grad is not None:
                o_grad = o_grad * self.momentum + param.grad / torch.sum(torch.abs(param.grad))
        super().zero_grad()

class WrapOptim: 
    @torch.no_grad()
    def __init__(self, params, epsilon, optimizer:Optimizer, min_value=0, max_value=1):
        self.optim = optimizer
        self.params = params
        self.epsilon = epsilon
        self.min_value = min_value
        self.max_value = max_value
        self.params_init = []
        for param in params:
            self.params_init.append(param.clone())

    @torch.no_grad()
    def step(self):
        self.optim.step()
        for param, param_init in zip(self.params, self.params_init):
            total_update = param - param_init
            update = torch.clamp(total_update, -self.epsilon, self.epsilon)

            param += update - total_update
            param.clamp_(self.min_value, self.max_value)
    
    def zero_grad(self):
        self.optim.zero_grad()

def get_optim(name, params, epsilon, **kwargs) -> I_FGSM:
    if name == 'i-fgsm': 
        return I_FGSM(params, epsilon, **kwargs)
    if name == 'mi-fgsm':
        return MI_FGSM(params, epsilon, **kwargs)

    optimizer = None
    if name == 'adadelta':
        optimizer = Adadelta(params)
    if name == 'adagrad':
        optimizer = Adagrad(params)
    if name == 'adam':
        optimizer = Adam(params)
    if name == 'adamw':
        optimizer = AdamW(params)
    if name == 'adamax':
        optimizer = Adamax(params)
    if name == 'asgd':
        optimizer = ASGD(params)
    if name == 'rmsprop':
        optimizer = RMSprop(params, lr=0.004)
    if name == 'rprop':
        optimizer = Rprop(params)
    if name == 'sgd':
        optimizer = SGD(params)
    
    if optimizer:
        return WrapOptim(params, epsilon, optimizer, **kwargs)

    return None
