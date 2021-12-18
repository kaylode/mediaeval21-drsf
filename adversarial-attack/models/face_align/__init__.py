from .base import BaseAlignment
from .fan import FANAlignment

alignment_factory = {"fan": FANAlignment}


def get_model(name: str):
    if name in alignment_factory:
        return alignment_factory[name]()
    else:
        raise ValueError("Unknown face detector model: {}".format(name))

