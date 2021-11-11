import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base Detector abstract class
    """

    def __init__(self):
        super().__init__()

    def preprocess(self, cv2_image):
        """
        Preprocess the input image before being passed into model
        :params:
            cv2_image: image in cv2 format. 
        :return: processed image
        """
        raise NotImplementedError("This is an interface method")

    def postprocess(self, adv_image):
        """
        Postprocess the adversarial image after being attacked.
        Convert the adversarial image into cv2 format
        :params:
            adv_image: attacked image. 
        :return: cv2 image
        """
        raise NotImplementedError("This is an interface method")

    def forward(self, adv_image, targets):
        """
        Forward the attacking image and targets to compute gradients
        :params:
            adv_image: adversarial image, also stores gradients. 
            targets: targets fit model and adversarial image. 
        :return: adversarial image
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

    def make_targets(self, predictions, cv2_image):
        """
        Make the targets from the predictions of model
        :params:
            predictions: model prediction. 
            cv2_image: raw cv2 image. 
        :return: model targets
        """
        raise NotImplementedError("This is an interface method")

    def get_face_box(self, predictions, return_probs):
        """
        Extract the bounding box from model predictions
        :params:
            predictions: model prediction. 
            return_probs: whether to return probability scores of face detection 
        :return: face bounding box in the image. In (x1,y1,x2,y2) format
        """
        raise NotImplementedError("This is an interface method")
