import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import util
import cv2

plt.ion()

plt.clf()

kernel_size = 5
low_threshold = 50
high_threshold = 150
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1

image = mpimg.imread('test_images/solidWhiteRight.jpg')

grayed = util.grayscale(image)

blurred = util.gaussian_blur(grayed, kernel_size)

edged = util.canny(blurred, low_threshold, high_threshold)

# plt.imshow(edged, cmap='gray')

mask = np.array([[[0, image.shape[0]],
                  [image.shape[1], image.shape[0]],
                  [image.shape[1]*0.5, image.shape[0]*0.50]]]).astype(int)

masked = util.region_of_interest(edged, mask)

detected, lines = util.hough_lines(masked, rho, theta, threshold, min_line_length, max_line_gap)
lines2 = cv2.HoughLines(masked, rho, theta, threshold, min_line_length, max_line_gap)


# plt.imshow(detected)

weighted = util.weighted_img(detected, image)

# plt.imshow(weighted)

slopes = (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2]-lines[:,0,0])
weights = np.sqrt((lines[:,0,2]-lines[:,0,0])**2+(lines[:,0,3]-lines[:,0,1])**2)
l_idx = np.logical_and(np.isfinite(slopes), slopes<0)
r_idx = np.logical_and(np.isfinite(slopes), slopes>0)
idx = np.logical_or(l_idx, r_idx)
l_m = np.average(slopes[l_idx], weights=weights[l_idx])
r_m = np.average(slopes[r_idx], weights=weights[r_idx])
l_b = np.average(lines[l_idx,0,1]-slopes[l_idx]*lines[l_idx,0,0], weights=weights[l_idx])
r_b = np.average(lines[r_idx,0,1]-slopes[r_idx]*lines[r_idx,0,0], weights=weights[r_idx])

l_0 = (int((image.shape[0]-l_b)/l_m), int(image.shape[0]))
l_1 = (int((image.shape[0]*0.5-l_b)/l_m), int(image.shape[0]*0.5))
r_0 = (int((image.shape[0]-r_b)/r_m), int(image.shape[0]))
r_1 = (int((image.shape[0]*0.5-r_b)/r_m), int(image.shape[0]*0.5))

lined = cv2.line(cv2.line(image, l_0, l_1, (255, 0, 0), 5), r_0, r_1, (255, 0, 0), 5)

plt.imshow(lined)

# hist, bins = np.histogram(slopes[idx], bins=100)
# width = 0.7*(bins[1] - bins[0])
# center = (bins[:-1] + bins[1:])/2
# plt.bar(center, hist, align='center', width=width)
