from .face3d import Face3DModel
from .gaze import GazeModel

gaze_factory = {"GazeModel": GazeModel, "Face3DModel": Face3DModel}


def get_model(name: str, **kwargs):
    if name in gaze_factory:
        return gaze_factory[name](**kwargs)
    else:
        raise ValueError("Unknown gaze detector model: {}".format(name))

