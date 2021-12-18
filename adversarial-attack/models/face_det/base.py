import torch.nn as nn

class BaseDetector(nn.Module):
    """
    Base Detector abstract class
    """
    def __init__(self):
        super().__init__()

    def preprocess(self, images):
        """
        Preprocess the input image before being passed into model
        :params:
            images: images in cv2 format. 
        :return: processed image
        """
        raise NotImplementedError("This is an interface method")

    def postprocess(self, adv_images):
        """
        Postprocess the adversarial image after being attacked.
        Convert the adversarial image into cv2 format
        :params:
            adv_images: attacked images. 
        :return: cv2 image
        """
        raise NotImplementedError("This is an interface method")

    def forward(self, adv_images, targets):
        """
        Forward the attacking image and targets to compute gradients
        :params:
            adv_images: adversarial images, also stores gradients. 
            targets: targets fit model and adversarial image. 
        :return: adversarial images
        """
        raise NotImplementedError("This is an interface method")

    def detect(self, query_input):
        """
        Model inference on the processed input
        :params:
            query_input: processed input. 
        :return: model predictions
        """
        raise NotImplementedError("This is an interface method")

    def make_targets(self, predictions, images):
        """
        Make the targets from the predictions of model
        :params:
            predictions: model prediction. 
            images: list of cv2 image. 
        :return: model targets
        """
        raise NotImplementedError("This is an interface method")

    def get_face_boxes(self, predictions, return_probs):
        """
        Extract the bounding box from model predictions
        :params:
            predictions: model prediction. 
            return_probs: whether to return probability scores of face detection 
        :return: face bounding box in the image. In (x1,y1,x2,y2) format
        """
        raise NotImplementedError("This is an interface method")