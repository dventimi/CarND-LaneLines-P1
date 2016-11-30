import util

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Turn Matplotlib interactive mode 'on' in order to see plots
plt.ion()

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)  #call as plt.imshow(gray, cmap='gray') to show a grayscaled image

import os
print(os.listdir("test_images/"))

plt.imshow(util.grayscale(image), cmap='gray')

