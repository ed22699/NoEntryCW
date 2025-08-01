import pandas as pd
import seaborn as sns
from houghSpace import HoughSpace
from sobelEdgeDetection import SobelEdgeDetection
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

imageName = "./No_entry/NoEntry6.bmp"
img = cv2.imread(imageName)
sobel = SobelEdgeDetection(img)
houghCirc = HoughSpace(img, minRadii=6, maxRadii=107, increment=1)
houghCirc.generateHough(T=100)

matrix = houghCirc.hough
# x = data[:, 0]
# y = data[:, 1]
# z = data[:, 2]

# Assume hough_3d is your 3D Hough Space with shape (height, width, num_radii)
# Compress to 2D by summing over the radius dimension
hough_2d_sum = np.sum(matrix, axis=2)

# Alternatively, compress using the maximum value across radii
hough_2d_max = np.max(matrix, axis=2)

# Visualize the 2D Hough Space
plt.imshow(hough_2d_sum, cmap='hot', interpolation='nearest')
plt.title('2D Hough Space (Summed Across Radii)')
plt.colorbar(label='Votes')
plt.show()
