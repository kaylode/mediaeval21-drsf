from typing import Any

import cv2
import torchvision.transforms as T
from omegaconf import DictConfig
import torch

def create_transform(config: DictConfig, is_tensor: bool = False) -> Any:
    if config.mode == "MPIIGaze":
        return T.ToTensor()
    elif config.mode == "MPIIFaceGaze":
        return _create_mpiifacegaze_transform(config, is_tensor)
    elif config.mode == "ETH-XGaze":
        return _create_ethxgaze_transform(config, is_tensor)
    else:
        raise ValueError


def _create_mpiifacegaze_transform(config: DictConfig, is_tensor: bool = False) -> Any:
    size = tuple(config.gaze_estimator.image_size)

    if is_tensor:
        transform = T.Compose(
            [
                T.Lambda(lambda x: x.flip(-3)),
                T.Lambda(lambda x: torch.nn.functional.interpolate(x, size=size, mode='bilinear')),
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),  # BGR
            ]
        )

    else:
        transform = T.Compose(
            [
                T.Lambda(lambda x: cv2.resize(x, size)),
                T.Lambda(lambda x: x[:, :, ::-1].copy()),  # RGB -> BGR 
                T.ToTensor(),
                T.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),  # BGR
            ]
        )
    return transform


def _create_ethxgaze_transform(config: DictConfig, is_tensor: bool = False) -> Any:
    size = tuple(config.gaze_estimator.image_size)
    
    if is_tensor:
        transform = T.Compose(
            [
                T.Lambda(lambda x: torch.nn.functional.interpolate(x, size=size, mode='bilinear')),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Lambda(lambda x: cv2.resize(x, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB
            ]
        )
    return transform
