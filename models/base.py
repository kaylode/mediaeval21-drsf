import torch.nn as nn

class BaseDetector(nn.Module):
    """
    Base Detector abstract class
    """
    def __init__(self):
        super().__init__()

    def preprocess(self):
        raise NotImplementedError("This is an interface method")

    def postprocess(self):
        raise NotImplementedError("This is an interface method")

    def forward(self):
        raise NotImplementedError("This is an interface method")

    def detect(self, x):
        raise NotImplementedError("This is an interface method")

    def make_targets(self):
        raise NotImplementedError("This is an interface method")

    def get_face_box(self):
        raise NotImplementedError("This is an interface method")