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
