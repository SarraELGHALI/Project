#!/usr/local/bin/python3
import os,sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# Load pickled data
import pickle
import random
import sys

import matplotlib.pyplot as plt
import numpy as np

import cv2
from IPython import get_ipython

#magic function 
ipy = get_ipython()
if ipy is not None : 
    ipy.run_line_magic('matplotlib','inline')


def load_data(training_file, testing_file): 
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']
    return ( X_train, y_train, X_test, y_test )

def show_imges(X_train,y_train): 
    # show image of 10 random data points
    fig, axs = plt.subplots(2,5, figsize=(15, 6))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    for i in range(10):
        index = random.randint(0, len(X_train))
        image = X_train[index]
        axs[i].axis('off')
        axs[i].imshow(image)
        axs[i].set_title(y_train[index])
    plt.show() 

def histogram_plt(y_train): 
    hist, bins = np.histogram(y_train, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

# Step 2: Design and Test a Model Architecture
### Preprocess the data here.
def convert_grascale(X_train,X_test): 
    X_train_rgb = X_train
    X_train_gry = np.sum(X_train/3, axis=3, keepdims=True)
    X_test_rgb = X_test
    X_test_gry = np.sum(X_test/3, axis=3, keepdims=True)
    return (X_train_rgb, X_train_gry, X_test_rgb, X_test_gry)




def normalize_data(X_train, X_test):
    X_train_normalized = (X_train - 128)/128 
    X_test_normalized = (X_test - 128)/128
    return (X_train_normalized, X_test_normalized) 

# Converting to grayscale - This worked well for Sermanet and LeCun as described in their traffic sign classification article. It also helps to reduce training time, which was nice when a GPU wasn't available.
def random_translate(img):
    rows,cols,_ = img.shape
    
    # allow translation up to px pixels in x and y directions
    px = 2
    dx,dy = np.random.randint(-px,px,2)

    M = np.float32([[1,0,dx],[0,1,dy]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst 
def random_scaling(img):   
    rows,cols,_ = img.shape

    # transform limits
    px = np.random.randint(-2,2)

    # ending locations
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])

    # starting locations (4 corners)
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    dst = cv2.warpPerspective(img,M,(rows,cols))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_warp(img):
    
    rows,cols,_ = img.shape

    # random scaling coefficients
    rndx = np.random.rand(3) - 0.5
    rndx *= cols * 0.06   # this coefficient determines the degree of warping
    rndy = np.random.rand(3) - 0.5
    rndy *= rows * 0.06

    # 3 starting points for transform, 1/4 way from edges
    x1 = cols/4
    x2 = 3*cols/4
    y1 = rows/4
    y2 = 3*rows/4

    pts1 = np.float32([[y1,x1],
                       [y2,x1],
                       [y1,x2]])
    pts2 = np.float32([[y1+rndy[0],x1+rndx[0]],
                       [y2+rndy[1],x1+rndx[1]],
                       [y1+rndy[2],x2+rndx[2]]])

    M = cv2.getAffineTransform(pts1,pts2)

    dst = cv2.warpAffine(img,M,(cols,rows))
    
    dst = dst[:,:,np.newaxis]
    
    return dst

def random_brightness(img):
    shifted = img + 1.0   # shift to (0,2) range
    img_max_value = max(shifted.flatten())
    max_coef = 2.0/img_max_value
    min_coef = max_coef - 0.1
    coef = np.random.uniform(min_coef, max_coef)
    dst = shifted * coef - 1.0
    return dst

def comparaision_data(input_indice,X_train_normalized, y_train): 
    # show comparisons of 5 random augmented data points
    choices = list(range(len(input_indices)))
    picks = []
    for i in range(5):
        rnd_index = np.random.randint(low=0,high=len(choices))
        picks.append(choices.pop(rnd_index))
    fig, axs = plt.subplots(2,5, figsize=(15, 6))
    fig.subplots_adjust(hspace = .2, wspace=.001)
    axs = axs.ravel()
    for i in range(5):
        image = X_train_normalized[input_indices[picks[i]]].squeeze()
        axs[i].axis('off')
        axs[i].imshow(image, cmap = 'gray')
        axs[i].set_title(y_train[input_indices[picks[i]]])
    for i in range(5):
        image = X_train_normalized[output_indices[picks[i]]].squeeze()
        axs[i+5].axis('off')
        axs[i+5].imshow(image, cmap = 'gray')
        axs[i+5].set_title(y_train[output_indices[picks[i]]])



if __name__=='__main__':
    training_file = "../traffic_sign_datasets/train.p"
    testing_file = "../traffic_sign_datasets/test.p"
    ( X_train, y_train, X_test, y_test ) = load_data(training_file, testing_file)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # Number of training examples
    n_train = len(X_train)

    # Number of testing examples.
    n_test = len(X_test)

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape

    # How many unique classes/labels there are in the dataset.
    n_classes = len(np.unique(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    show_imges(X_train,y_train)
    histogram_plt(y_train)
    
    # convert to grascale 
    (X_train_rgb, X_train_gry, X_test_rgb, X_test_gry) = convert_grascale(X_train, X_test)
    print('RGB shape:', X_train_rgb.shape)
    print('Grayscale shape:', X_train_gry.shape)

    X_train = X_train_gry
    X_test = X_test_gry

    print('done') 

    # Visualize rgb vs grayscale
    n_rows = 8
    n_cols = 10
    offset = 9000
    fig, axs = plt.subplots(n_rows,n_cols, figsize=(18, 14))
    fig.subplots_adjust(hspace = .1, wspace=.001)
    axs = axs.ravel()
    for j in range(0,n_rows,2):
        for i in range(n_cols):
            index = i + j*n_cols
            image = X_train_rgb[index + offset]
            axs[index].axis('off')
            axs[index].imshow(image)
        for i in range(n_cols):
            index = i + j*n_cols + n_cols 
            image = X_train_gry[index + offset - n_cols].squeeze()
            axs[index].axis('off')
            axs[index].imshow(image, cmap='gray')

    
    plt.show()

    print(np.mean(X_train))
    print(np.mean(X_test))
    ## Normalize the train and test datasets to (-1,1)
    (X_train_normalized, X_test_normalized) = normalize_data(X_train, X_test)
    print(np.mean(X_train_normalized))
    print(np.mean(X_test_normalized))
    print("Original shape:", X_train.shape)
    print("Normalized shape:", X_train_normalized.shape)
    fig, axs = plt.subplots(1,2, figsize=(10, 3))
    axs = axs.ravel()

    axs[0].axis('off')
    axs[0].set_title('normalized')
    axs[0].imshow(X_train_normalized[0].squeeze(), cmap='gray')

    axs[1].axis('off')
    axs[1].set_title('original')
    axs[1].imshow(X_train[0].squeeze(), cmap='gray')    


    test_img = X_train_normalized[22222]

    test_dst = random_translate(test_img)

    fig, axs = plt.subplots(1,2, figsize=(10, 3))

    axs[0].axis('off')
    axs[0].imshow(test_img.squeeze(), cmap='gray')
    axs[0].set_title('original')

    axs[1].axis('off')
    axs[1].imshow(test_dst.squeeze(), cmap='gray')
    axs[1].set_title('translated')

    print('shape in/out:', test_img.shape, test_dst.shape)

    # histogram of label frequency (once again, before data augmentation)
    hist, bins = np.histogram(y_train, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()
    print(np.bincount(y_train))
    print("minimum samples for any label:", min(np.bincount(y_train)))
    print('X, y shapes:', X_train_normalized.shape, y_train.shape)

    input_indices = []
    output_indices = []

    for class_n in range(0, n_classes):
        print(class_n, ': ', end='')
        class_indices = np.where(y_train == class_n)
        n_samples = len(class_indices[0])
        if n_samples < 800:
            for i in range(800 - n_samples):
                input_indices.append(class_indices[0][i%n_samples])
                output_indices.append(X_train_normalized.shape[0])
                new_img = X_train_normalized[class_indices[0][i % n_samples]]
                new_img = random_translate(random_scaling(random_warp(random_brightness(new_img))))
                X_train_normalized = np.concatenate((X_train_normalized, [new_img]), axis=0)
                y_train = np.concatenate((y_train, [class_n]), axis=0)
                if i % 50 == 0:
                    print('|', end='')
                elif i % 10 == 0:
                    print('-',end='')
            print('')
            
    print('X, y shapes:', X_train_normalized.shape, y_train.shape)
    comparaision_data(input_indices,X_train_normalized, y_train)

    # histogram of label frequency

    hist, bins = np.histogram(y_train, bins=n_classes)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()