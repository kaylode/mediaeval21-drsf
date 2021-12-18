from .face3d import Face3DModel
from .gaze import GazeModel

gaze_factory = {"GazeModel": GazeModel, "Face3DModel": Face3DModel}


def get_model(model_name: str, **kwargs):
    if model_name in gaze_factory:
        return gaze_factory[model_name](**kwargs)
    else:
        raise ValueError("Unknown gaze detector model: {}".format(model_name))

