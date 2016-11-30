import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import util

plt.ion()

kernel_size = 5
low_threshold = 50
high_threshold = 150
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1

image = mpimg.imread('test_images/solidWhiteRight.jpg')

gray = util.grayscale(image)

blur = util.gaussian_blur(gray, kernel_size)

canny = util.canny(blur, low_threshold, high_threshold)

# plt.imshow(canny, cmap='gray')

mask = np.array([[[0, image.shape[0]],
                  [image.shape[1], image.shape[0]],
                  [image.shape[1]*0.5, image.shape[0]*0.59]]]).astype(int)

masked = util.region_of_interest(canny, mask)

lines = util.hough_lines(masked, rho, theta, threshold, min_line_length, max_line_gap)

plt.imshow(lines)

weighted = util.weighted_img(lines, image)

# plt.imshow(weighted)

# Iterate over the output "lines" and draw lines on the blank
# util.draw_lines(blur_gray, lines)        

# # Create a "color" binary image to combine with line image
# color_edges = np.dstack((edges, edges, edges)) 

# # Draw the lines on the edge image
# combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0) 
# plt.imshow(combo)
