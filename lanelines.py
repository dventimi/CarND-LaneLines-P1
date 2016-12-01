from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import util
import cv2

plt.ion()

plt.clf()

image = mpimg.imread('test_images/solidWhiteRight.jpg')
         
def process_image(image):
    kernel_size = 5
    low_threshold = 50
    high_threshold = 150
    rho = 1
    theta = np.pi/180
    threshold = 1
    min_line_length = 10
    max_line_gap = 1
    
    grayed = util.grayscale(image)
    blurred = util.gaussian_blur(grayed, kernel_size)
    edged = util.canny(blurred, low_threshold, high_threshold)
    mask = np.array([[[0, image.shape[0]],
                      [image.shape[1], image.shape[0]],
                      [image.shape[1]*0.5, image.shape[0]*0.50]]]).astype(int)

    masked = util.region_of_interest(edged, mask)
    detected, lines = util.hough_lines(masked, rho, theta, threshold, min_line_length, max_line_gap)
    lines2 = cv2.HoughLines(masked, rho, theta, threshold, min_line_length, max_line_gap)
    weighted = util.weighted_img(detected, image)
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
    return lined

lined = process_image(image)

plt.imshow(lined)

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)


