from .base import BaseDetector
from .mtcnn import MTCNNDetector
from .retinaface import RetinaFaceDetector

detector_factory = {"mtcnn": MTCNNDetector, "retinaface": RetinaFaceDetector}


def get_model(name: str):
    if name in detector_factory:
        return detector_factory[name]()
    else:
        raise ValueError("Unknown face detector model: {}".format(name))

