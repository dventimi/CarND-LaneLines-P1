# Import useful packages.
from IPython.display import HTML
from moviepy.editor import VideoFileClip
import cv2
import glob
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import util

# Define function y=f(x,m,b) and inverse function x=g(y,m,b).
y = lambda x,m,b: m*x+b
x = lambda y,m,b: (y-b)/m

# Define useful image features as functions.
top = lambda img: 0
bottom = lambda img: img.shape[0]
left = lambda img: 0
right = lambda img: img.shape[1]
width = lambda img: right(img) - left(img)
height = lambda img: bottom(img) - top(img)
horizon = lambda img: int(img.shape[0]*0.60)
centerline = lambda img: int(img.shape[1]*0.5)
center = lambda img: [horizon(img), centerline(img)]
ground = lambda img: np.array([[[horizon(img), left(img)],
                                [horizon(img), right(img)],
                                [bottom(img), right(img)],
                                [bottom(img), left(img)]]])
sky = lambda img: np.array([[[top(img), left(img)],
                             [top(img), right(img)],
                             [bottom(img), right(img)],
                             [bottom(img), left(img)]]])
road = lambda img: np.array([[[horizon(img), centerline(img)-0.10*width(img)/2],
                              [horizon(img), centerline(img)+0.10*width(img)/2],
                              [bottom(img), centerline(img)+0.95*width(img)/2],
                              [bottom(img), centerline(img)-0.95*width(img)/2]]]).astype(int)
roadline_pts = lambda img,m,b: ((int(x(bottom(img),m,b)), bottom(img)),
                                (int(x(horizon(img),m,b)), horizon(img)))

# Define functions to get slopes and y-intercepts for an array of lines.
slope = lambda lines: (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2]-lines[:,0,0])
intercept = lambda lines, m: lines[:,0,1]-m*lines[:,0,0]

# Define functions get indices into lines array, for left line and for right line.
lidx = lambda slopes: np.logical_and(np.isfinite(slopes), slopes<0)
ridx = lambda slopes: np.logical_and(np.isfinite(slopes), slopes>0)

# Define wrapper functions that adapt parameters as keyward argument dictionaries.
def grayscale_image(img, **kwargs):
    return util.grayscale(img)

def blur_image(img, **kwargs):
    return util.gaussian_blur(img, kwargs['kernel_size'])

def edge_image(img, **kwargs):
    return util.canny(img, kwargs['low_threshold'], kwargs['high_threshold'])

def mask_image(img, vertices, **kwargs):
    return util.region_of_interest(img, vertices)

def detect_image(img, **kwargs):
    lines = cv2.HoughLinesP(img, kwargs['rho'], kwargs['theta']*np.pi/180, kwargs['threshold'], np.array([]), minLineLength=kwargs['min_line_length'], maxLineGap=kwargs['max_line_gap'])
    m = slope(lines)
    b = intercept(lines, m)
    image = util.hough_lines(img, kwargs['rho'], kwargs['theta']*np.pi/180, kwargs['threshold'], kwargs['min_line_length'], kwargs['max_line_gap'])
    return image, lines, m, b

def average_lines(img, lines, m, b, **kwargs):
    image = np.copy(img)
    mbar = [np.mean(m[lidx(m)]), np.mean(m[ridx(m)])]
    bbar = [np.mean(b[lidx(m)]), np.mean(b[ridx(m)])]
    l_pts = roadline_pts(img, mbar[0], bbar[0])
    r_pts = roadline_pts(img, mbar[1], bbar[1])
    cv2.line(image, l_pts[0], l_pts[1], (0, 255, 0), 5)
    cv2.line(image, r_pts[0], r_pts[1], (0, 255, 0), 5)
    return image

# Define processing pipeline as a function of wrapper and helper functions.
def process_image(img0):
    img1 = grayscale_image(img0, **theta)
    img2 = blur_image(img1, **theta)
    img3 = edge_image(img2, **theta)
    img4 = mask_image(img3, road(image)[:,:,::-1], **theta)
    img5, lines, slopes, intercepts = detect_image(img4, **theta)
    img6 = util.weighted_img(img5, img0)
    img7 = average_lines(img6, lines, slopes, intercepts, **theta)
    return img7

# Define tuning parameters theta as a global variable.
theta = {'kernel_size':5,
         'low_threshold':50,
         'high_threshold':150,
         'rho':2,
         'theta':1,
         'threshold':70,
         'min_line_length':3,
         'max_line_gap':1}

# Process images in the "test_images" directory.
for path in glob.glob('test_images/solid*.jpg'):
    fname = path.split("/")[1]
    image = mpimg.imread(path)
    processed_image = process_image(image)
    mpimg.imsave("test_images/processed_%s" % fname, processed_image)

# Process first test video.
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

# Process second test video.
white_output = 'yellow.mp4'
clip1 = VideoFileClip("solidYellowLeft.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
