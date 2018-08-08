#!/usr/local/bin/python3

# importing librairies 
import os,sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import matplotlib.pyplot as plt 
import cv2 
import numpy as np 
from moviepy.editor import VideoFileClip
from IPython import get_ipython
import glob  

#magic function 
ipy = get_ipython()
if ipy is not None : 
    ipy.run_line_magic('matplotlib','inline')

# dislplay images 
def show_images(images, cmap=None): 
    clons = 2 
    rows = (len(images)+1)//clons
    plt.figure(figsize=(10,11))
    for i, image in enumerate(images): 
        plt.subplot(rows,clons,i+1)
        # use gray scale color map if there is only one channel 
        cmap = 'gray' if len(image.shape )==2 else cmap
        plt.imshow(image,cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show() 


# clor selection 
# image is expected be in RGB color space
def select_rgb_white_yellow(image): 
    # white color mask
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)
    # yellow color mask
    lower = np.uint8([190, 190,   0])
    upper = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked
def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)


def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def select_white_yellow(image):
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)
