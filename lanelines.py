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

# Udacity helper functions
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

# Define function y=f(x,m,b) and inverse function x=g(y,m,b).
y = lambda x,m,b: m*x+b
x = lambda y,m,b: (y-b)/m

# Define useful image features as functions.
top = lambda img: 0
bottom = lambda img: int(img.shape[0])
hood = lambda img: bottom(img)*(1-theta['hood'])
left = lambda img: 0
right = lambda img: img.shape[1]
width = lambda img: right(img) - left(img)
height = lambda img: bottom(img) - top(img)
horizon = lambda img: int(img.shape[0]*theta['horizon'])
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
trapezoid = lambda img: np.array([[[horizon(img), centerline(img)-theta['trapezoid_top_factor']*width(img)/2],
                                   [horizon(img), centerline(img)+theta['trapezoid_top_factor']*width(img)/2],
                                   [hood(img), centerline(img)+theta['trapezoid_bottom_factor']*width(img)/2],
                                   [hood(img), centerline(img)-theta['trapezoid_bottom_factor']*width(img)/2]]]).astype(int)
trapezoid_pts = lambda img,m,b: ((int(x(hood(img),m,b)), int(hood(img))),
                                 (int(x(horizon(img),m,b)), int(horizon(img))))

# Define functions to get slopes and y-intercepts for an array of lines.
slope = lambda lines: (lines[:,0,3]-lines[:,0,1])/(lines[:,0,2]-lines[:,0,0])
intercept = lambda lines, m: lines[:,0,1]-m*lines[:,0,0]

# Define functions get indices into lines array, for left line and for right line.
lidx = lambda slopes: np.logical_and(np.isfinite(slopes), slopes<0, np.abs(slopes)>math.tan(theta['angle_cutoff']))
ridx = lambda slopes: np.logical_and(np.isfinite(slopes), slopes>0, np.abs(slopes)>math.tan(theta['angle_cutoff']))

# Define wrapper functions that adapt the Udacity helper functions.
def grayscale_image(img):
    return grayscale(img)

def blur_image(img):
    return gaussian_blur(img, theta['kernel_size'])

def edge_image(img):
    return canny(img, theta['low_threshold'], theta['high_threshold'])

def mask_image(img, vertices):
    return region_of_interest(img, vertices)

def detect_image(img):
    lines = cv2.HoughLinesP(img, theta['rho'], theta['theta']*np.pi/180, theta['threshold'], np.array([]), minLineLength=theta['min_line_length'], maxLineGap=theta['max_line_gap'])
    m = slope(lines)
    b = intercept(lines, m)
    image = hough_lines(img, theta['rho'], theta['theta']*np.pi/180, theta['threshold'], theta['min_line_length'], theta['max_line_gap'])
    return image, lines, m, b

def average_lines(img, lines, m, b):
    image = np.copy(img)
    mbar = [np.mean(m[lidx(m)]), np.mean(m[ridx(m)])]
    bbar = [np.mean(b[lidx(m)]), np.mean(b[ridx(m)])]
    l_pts = trapezoid_pts(img, mbar[0], bbar[0])
    r_pts = trapezoid_pts(img, mbar[1], bbar[1])
    cv2.line(image, l_pts[0], l_pts[1], (0, 255, 0), 5)
    cv2.line(image, r_pts[0], r_pts[1], (0, 255, 0), 5)
    return image

def detect_lines(img):
    img1 = grayscale_image(img0)
    img2 = blur_image(img1)
    img3 = edge_image(img2)
    img4 = mask_image(img3, trapezoid(img3)[:,:,::-1])
    img5, lines, slopes, intercepts = detect_image(img4)

# Top-Level functions

def process_image1(img0):
    img1 = grayscale_image(img0)
    img2 = blur_image(img1)
    img3 = edge_image(img2)
    img4 = mask_image(img3, trapezoid(img3)[:,:,::-1])
    img5, lines, slopes, intercepts = detect_image(img4)
    img6 = weighted_img(img5, img0)
    return img6

def process_image2(img0):
    img1 = grayscale_image(img0)
    img2 = blur_image(img1)
    img3 = edge_image(img2)
    img4 = mask_image(img3, trapezoid(img3)[:,:,::-1])
    img5, lines, slopes, intercepts = detect_image(img4)
    img6 = average_lines(img0, lines, slopes, intercepts)
    return img6

# Parameters for part 1

theta = {'horizon':0.60,
         'hood':0.0,
         'trapezoid_top_factor':0.10,
         'trapezoid_bottom_factor':0.90,
         'angle_cutoff':0.45,
         'kernel_size':5,
         'low_threshold':50,
         'high_threshold':150,
         'rho':2,
         'theta':1,
         'threshold':40,
         'min_line_length':3,
         'max_line_gap':1}

# Process images in the "test_images" directory.
for path in glob.glob('test_images/solid*.jpg'):
    fname = path.split("/")[1]
    image = mpimg.imread(path)
    processed_image = process_image1(image)
    mpimg.imsave("test_images/processed_%s" % fname, processed_image)

# Process first test video.
white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image2)
white_clip.write_videofile(white_output, audio=False)

# Process second test video.
yellow_output = 'yellow.mp4'
clip1 = VideoFileClip("solidYellowLeft.mp4")
yellow_clip = clip1.fl_image(process_image2)
yellow_clip.write_videofile(yellow_output, audio=False)

# Parameters for part 2 (challenge)

theta = {'horizon':0.61,
         'hood':0.07,
         'trapezoid_top_factor':0.10,
         'trapezoid_bottom_factor':0.90,
         'angle_cutoff':0.75,
         'kernel_size':5,
         'low_threshold':50,
         'high_threshold':150,
         'rho':2,
         'theta':1,
         'threshold':30,
         'min_line_length':3,
         'max_line_gap':1}

# Process challenge video.
challenge_output = 'extra.mp4'
clip1 = VideoFileClip("challenge.mp4")
challenge_clip = clip1.fl_image(process_image2)
challenge_clip.write_videofile(challenge_output, audio=False)
