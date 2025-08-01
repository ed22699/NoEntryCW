import numpy as np
import cv2


class SobelEdgeDetection:
    def __init__(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = image
        self.magImage = np.zeros(
            [image.shape[0], image.shape[1]], dtype=np.float32)
        self.phaseImage = np.zeros(
            [image.shape[0], image.shape[1]], dtype=np.float32)
        self.__processImg()

    def __processImg(self):
        image = self.image
        # we need to create a padded version of the input
        # or there will be border effects
        xKernelRot = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        yKernelRot = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        kernelRadiusX = round((yKernelRot.shape[0] - 1) / 2)
        kernelRadiusY = round((yKernelRot.shape[1] - 1) / 2)

        paddedInput = cv2.copyMakeBorder(image,
                                         kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                                         cv2.BORDER_REPLICATE)

        # now we can do the convoltion
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                Patch = paddedInput[i:i+xKernelRot.shape[0],
                                    j:j+xKernelRot.shape[1]]
                xSum = (np.multiply(Patch, xKernelRot)).sum()
                ySum = (np.multiply(Patch, yKernelRot)).sum()

                self.magImage[i, j] = np.sqrt(xSum**2 + ySum**2)
                phaseVal = np.arctan2(ySum, xSum)
                self.phaseImage[i, j] = phaseVal * (180 / np.pi)

    def getMagnitudeImage(self):
        # magImg = cv2.convertScaleAbs(
        #     self.magImage, alpha=255/self.magImage.max())
        # # Normalize the magnitude image to 0-255
        magImg = cv2.normalize(self.magImage, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return magImg.astype(np.uint8)

    def getPhaseImage(self):
        return self.phaseImage
