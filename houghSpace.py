from sobelEdgeDetection import SobelEdgeDetection
import cv2
import numpy as np


class HoughSpace:
    def __init__(self, image, maxRadii=50, minRadii=5, increment=1):
        self.image = image
        self.maxRadii = maxRadii
        self.minRadii = minRadii
        self.increment = increment
        self.hough = np.zeros((*image.shape[:2], maxRadii - minRadii))

    def __GaussianBlur(self, image, size):
        # intialise the output using the input
        blurredOutput = np.zeros(
            [image.shape[0], image.shape[1]], dtype=np.float32)
        # create the Gaussian kernel in 1D
        kX = cv2.getGaussianKernel(size, 1)
        kY = cv2.getGaussianKernel(size, 1)
        # make it 2D multiply one by the transpose of the other
        kernel = kX * kY.T

        # we need to create a padded version of the input
        # or there will be border effects
        kernelRadiusX = round((kernel.shape[0] - 1) / 2)
        kernelRadiusY = round((kernel.shape[1] - 1) / 2)

        paddedInput = cv2.copyMakeBorder(image,
                                         kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                                         cv2.BORDER_REPLICATE)

        # now we can do the convoltion
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                patch = paddedInput[i:i+kernel.shape[0], j:j+kernel.shape[1]]
                total = (np.multiply(patch, kernel)).sum()
                # set the output value as the sum of the convolution
                blurredOutput[i, j] = total

        return blurredOutput

    def generateHough(self, T=100):
        # get binary edge
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = self.__GaussianBlur(gray, 5)
        edges = SobelEdgeDetection(blur)
        ret, binaryEdge = cv2.threshold(
            edges.getMagnitudeImage(), T, 255, cv2.THRESH_BINARY)
        # get phaseImage
        phaseImage = edges.getPhaseImage()

        edgePoints = np.argwhere(binaryEdge > 0)
        for y, x in edgePoints:
            r = 0
            angle = np.radians(phaseImage[y][x])
            while r < self.maxRadii - self.minRadii:
                rad = self.minRadii + r
                xRad = int(rad * np.cos(angle))
                yRad = int(rad * np.sin(angle))
                b = x - xRad
                a = y - yRad
                if (a >= 0 and a < self.hough.shape[0] and b >= 0 and b < self.hough.shape[1]):
                    self.hough[a, b, r] += 1
                b = x + xRad
                a = y + yRad
                if (a >= 0 and a < self.hough.shape[0] and b >= 0 and b < self.hough.shape[1]):
                    self.hough[a, b, r] += 1
                r += self.increment

    def drawCircles(self, T=5):
        circles = np.argwhere(self.hough > T)
        self.circles = circles
        for y, x, r in circles:
            cv2.circle(self.image, (x, y), r + self.minRadii, (255, 0, 0), 2)
        return self.image

    def getBoxes(self, T=5):
        circles = np.argwhere(self.hough > T)
        squares = []
        for circle in circles:
            square = []
            rad = circle[2] + self.minRadii
            # x coord
            square.append(circle[1] - rad)
            # y coord
            square.append(circle[0] - rad)
            # width and height
            square.append(2 * rad)
            square.append(2 * rad)
            squares.append(square)
        return squares
