import numpy as np
import cv2

class Pixelate:
    def __init__(self, blocks=3) -> None:
        self.blocks = blocks

    def forward(self, image, face_box):
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

class Blur:
    def __init__(self, kernel_size=3) -> None:
        self.kernel_size = (kernel_size, kernel_size)

    def forward(self, image, face_box):
        x1,y1,x2,y2 = face_box
        crop = image[y1:y2, x1:x2, :]
        crop = cv2.blur(crop, self.kernel_size)
        image[y1:y2, x1:x2, :] = crop.copy()

        # return the blurred image
        return image