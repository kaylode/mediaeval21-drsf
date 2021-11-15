import os
import bz2
import logging
import operator
import pathlib

import cv2
import numpy as np
import torch.hub
import yaml
from omegaconf import DictConfig
from typing import Tuple

from .common.face_model import FaceModel
from .common.face_model_68 import FaceModel68
from .common.face_model_mediapipe import FaceModelMediaPipe

logger = logging.getLogger(__name__)

def get_3d_face_model(config: DictConfig) -> FaceModel:
    if config.face_detector.mode == "mediapipe":
        return FaceModelMediaPipe()
    else:
        return FaceModel68()


def download_dlib_pretrained_model() -> None:
    logger.debug("Called download_dlib_pretrained_model()")

    dlib_model_dir = pathlib.Path("~/.ptgaze/dlib/").expanduser()
    dlib_model_dir.mkdir(exist_ok=True, parents=True)
    dlib_model_path = dlib_model_dir / "shape_predictor_68_face_landmarks.dat"
    logger.debug(
        f"Update config.face_detector.dlib_model_path to {dlib_model_path.as_posix()}"
    )

    if dlib_model_path.exists():
        logger.debug(
            f"dlib pretrained model {dlib_model_path.as_posix()} already exists."
        )
        return

    logger.debug("Download the dlib pretrained model")
    bz2_path = dlib_model_path.as_posix() + ".bz2"
    torch.hub.download_url_to_file(
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2", bz2_path
    )
    with bz2.BZ2File(bz2_path, "rb") as f_in, open(dlib_model_path, "wb") as f_out:
        data = f_in.read()
        f_out.write(data)


def download_mpiigaze_model() -> pathlib.Path:
    logger.debug("Called _download_mpiigaze_model()")
    output_path = "demo/pretrained/mpiigaze_resnet_preact.pth"
    if not os.path.exists(output_path):
        logger.debug("Download the pretrained model")
        torch.hub.download_url_to_file(
            "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiigaze_resnet_preact.pth",
            output_path.as_posix(),
        )
    else:
        logger.debug(f"The pretrained model {output_path} already exists.")
    return output_path


def download_mpiifacegaze_model() -> pathlib.Path:
    logger.debug("Called _download_mpiifacegaze_model()")
    output_path = "demo/pretrained/mpiifacegaze_resnet_simple.pth"
    if not os.path.exists(output_path):
        logger.debug("Download the pretrained model")
        torch.hub.download_url_to_file(
            "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiifacegaze_resnet_simple.pth",
            output_path.as_posix(),
        )
    else:
        logger.debug(f"The pretrained model {output_path} already exists.")
    return output_path


def download_ethxgaze_model() -> pathlib.Path:
    logger.debug("Called _download_ethxgaze_model()")
    output_path = "demo/pretrained/eth-xgaze_resnet18.pth"
    if not os.path.exists(output_path):
        logger.debug("Download the pretrained model")
        torch.hub.download_url_to_file(
            "https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth",
            output_path.as_posix(),
        )
    else:
        logger.debug(f"The pretrained model {output_path} already exists.")
    return output_path


def generate_dummy_camera_params(width, height) -> str:

    dic = {
        "image_width": width,
        "image_height": height,
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [width, 0.0, width // 2, 0.0, width, height // 2, 0.0, 0.0, 1.0],
        },
        "distortion_coefficients": {
            "rows": 1,
            "cols": 5,
            "data": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    }
    with open("/tmp/camera_params.yaml", "w") as f:
        yaml.safe_dump(dic, f)

    return "/tmp/camera_params.yaml"
    


def _expanduser(path: str) -> str:
    if not path:
        return path
    return pathlib.Path(path).expanduser().as_posix()


def expanduser_all(config: DictConfig) -> None:
    if hasattr(config.face_detector, "dlib_model_path"):
        config.face_detector.dlib_model_path = _expanduser(
            config.face_detector.dlib_model_path
        )
    config.gaze_estimator.checkpoint = _expanduser(config.gaze_estimator.checkpoint)
    config.gaze_estimator.camera_params = _expanduser(
        config.gaze_estimator.camera_params
    )
    config.gaze_estimator.normalized_camera_params = _expanduser(
        config.gaze_estimator.normalized_camera_params
    )
    if hasattr(config.demo, "image_path"):
        config.demo.image_path = _expanduser(config.demo.image_path)
    if hasattr(config.demo, "video_path"):
        config.demo.video_path = _expanduser(config.demo.video_path)
    if hasattr(config.demo, "output_dir"):
        config.demo.output_dir = _expanduser(config.demo.output_dir)


def _check_path(config: DictConfig, key: str) -> None:
    path_str = operator.attrgetter(key)(config)
    path = pathlib.Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"config.{key}: {path.as_posix()} not found.")
    if not path.is_file():
        raise ValueError(f"config.{key}: {path.as_posix()} is not a file.")


def check_path_all(config: DictConfig) -> None:
    _check_path(config, "gaze_estimator.checkpoint")
    _check_path(config, "gaze_estimator.camera_params")
    _check_path(config, "gaze_estimator.normalized_camera_params")
    if config.demo.image_path:
        _check_path(config, "demo.image_path")
    if config.demo.video_path:
        _check_path(config, "demo.video_path")


def convert_to_unit_vector(
        angles: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pitches = angles[:, 0]
    yaws = angles[:, 1]
    x = -torch.cos(pitches) * torch.sin(yaws)
    y = -torch.sin(pitches)
    z = -torch.cos(pitches) * torch.cos(yaws)
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(predictions: torch.Tensor,
                        labels: torch.Tensor) -> torch.Tensor:
    pred_x, pred_y, pred_z = convert_to_unit_vector(predictions)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi