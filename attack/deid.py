import numpy as np
import cv2

class DeID:
    """
    Base class for deid function
    """
    def __init__(self) -> None:
        pass

    def forward_batch(self, images, face_boxes):
        """
        Forward batch of images
        """
        deid_images = []
        for (image, face_box) in zip(images, face_boxes):
            deid_images.append(self(image, face_box))
        return deid_images

class Pixelate(DeID):
    """
    Pixelate face in the image
    :params:
        blocks: number of pixelated blocks
    """
    def __init__(self, blocks=3) -> None:
        super().__init__()
        self.blocks = blocks

    def __call__(self, image, face_box):
        """
        :params:
            image: cv2 image
            face_box: bounding box of face. In (x1,y1,x2,y2) format
        """
        x1,y1,x2,y2 = face_box
        crop = image[y1:y2, x1:x2, :]
        
        # divide the input image into NxN blocks
        (h, w) = crop.shape[:2]
        xSteps = np.linspace(0, w, self.blocks + 1, dtype="int")
        ySteps = np.linspace(0, h, self.blocks + 1, dtype="int")
        # loop over the blocks in both the x and y direction
        for i in range(1, len(ySteps)):
            for j in range(1, len(xSteps)):
                # compute the starting and ending (x, y)-coordinates
                # for the current block
                startX = xSteps[j - 1]
                startY = ySteps[i - 1]
                endX = xSteps[j]
                endY = ySteps[i]
                # extract the ROI using NumPy array slicing, compute the
                # mean of the ROI, and then draw a rectangle with the
                # mean RGB values over the ROI in the original image
                roi = crop[startY:endY, startX:endX]

                (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
                cv2.rectangle(crop, (startX, startY), (endX, endY),
                (B, G, R), -1)
        
        image[y1:y2, x1:x2, :] = crop.copy()

        # return the pixelated blurred image
        return image

class Blur(DeID):
    """
    Gaussian Blur face in the image
    :params:
        blocks: Gaussian filter size
    """
    def __init__(self, kernel_size=3) -> None:
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size)

    def __call__(self, image, face_box):
        """
        :params:
            image: cv2 image
            face_box: bounding box of face. In (x1,y1,x2,y2) format
        """
        x1,y1,x2,y2 = face_box
        crop = image[y1:y2, x1:x2, :]
        crop = cv2.blur(crop, self.kernel_size)
        image[y1:y2, x1:x2, :] = crop.copy()

        # return the blurred image
        return image