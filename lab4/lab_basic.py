import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# function: check if an image is type RGB; change type to gray
def check_rgb2gray(img):
    if len(img.shape) == 3:  # if image is type RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


# function: plot gray image
def show_image(img_name, img):
    plt.imshow(img, cmap='gray')
    plt.title(img_name, fontsize=12)
    plt.xticks([])
    plt.yticks([])

