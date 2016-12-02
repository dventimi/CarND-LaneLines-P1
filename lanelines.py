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
road = lambda img:  np.array([[[horizon(img), centerline(img)-0.10*width(img)/2],
                               [horizon(img), centerline(img)+0.10*width(img)/2],
                               [bottom(img), centerline(img)+0.95*width(img)/2],
                               [bottom(img), centerline(img)-0.95*width(img)/2]]]).astype(int)

y = lambda x,m,b: m*x+b
x = lambda y,m,b: (y-b)/m

roadline_pts = lambda img,m,b: ((int(x(bottom(img),m,b)), bottom(img)),
                                (int(x(horizon(img),m,b)), horizon(img)))
slope = lambda lines:  (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2]-lines[:,0,0])
intercept = lambda lines, m:  lines[:,0,1]-m*lines[:,0,0]

lidx = lambda slopes:  np.logical_and(np.isfinite(slopes), slopes<0)
ridx = lambda slopes:  np.logical_and(np.isfinite(slopes), slopes>0)

def process_image(img0, kernel_size=5, low_threshold=50, high_threshold=150, rho=2, theta=1, threshold=50, min_line_length=3, max_line_gap=1):
    img1 = util.grayscale(img0)
    img2 = util.gaussian_blur(img1, kernel_size)
    img3 = util.canny(img2, low_threshold, high_threshold)
    img4 = util.region_of_interest(img3, road(img0)[:,:,::-1])
    img5, lines = util.hough_lines(img4, rho, theta*np.pi/180, threshold, min_line_length, max_line_gap)
    img6 = util.weighted_img(img5, img0)
    img7 = np.copy(img6)
    m = slope(lines)
    b = intercept(lines, m)
    mbar = [np.mean(m[lidx(m)]), np.mean(m[ridx(m)])]
    bbar = [np.mean(b[lidx(m)]), np.mean(b[ridx(m)])]
    l_pts = roadline_pts(img0, mbar[0], bbar[0])
    r_pts = roadline_pts(img0, mbar[1], bbar[1])
    cv2.line(img7, l_pts[0], l_pts[1], (0, 255, 0), 5)
    cv2.line(img7, r_pts[0], r_pts[1], (0, 255, 0), 5)
    return img7

for path in glob.glob('test_images/solid*.jpg'):
    fname = path.split("/")[1]
    image = mpimg.imread(path)
    processed_image = process_image(image)
    mpimg.imsave("test_images/processed_%s" % fname, processed_image)

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
