import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

# Keywords: unsharp masking


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


# function: Gaussian filter
def gaussian_filter(img, s):
    # s: sigma in Gaussian filter
    gk = get_gaussian_kernel(s)  # Gaussian kernel
    img = signal.convolve2d(img, gk, "same")
    return img


# function: get the Gaussian kernel
def get_gaussian_kernel(s):
    # s: sigma in Gaussian filter
    a = 6 * s + 1  # size of the kernel
    mid = 3 * s  # position of the kernel center
    k = np.zeros((a, a))
    c = math.exp(- pow(mid, 2) / pow(s, 2))  # center value
    for i in range(-mid, mid + 1):
        for j in range(-mid, mid + 1):
            k[mid + i, mid + j] = math.exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(s, 2))) / c  # pixel value
    k_sum = np.sum(k)
    k = k / k_sum
    return k


# function: unsharp masking using Gaussian filter(sigma = 3)
def unsharp_masking(img):
    img = 2 * img - gaussian_filter(img, 3)
    return img


image0 = np.load("lab2.npy")
image0 = check_rgb2gray(image0)
plt.subplot(1, 3, 1)
show_image("original image", image0)
print("image0 loaded")
print("========")

image1 = gaussian_filter(image0, 3)
plt.subplot(1, 3, 2)
show_image("Gaussian filtered image", image1)
print("image1 done")
print("========")

image2 = unsharp_masking(image0)
plt.subplot(1, 3, 3)
show_image("unsharp masked image", image2)
print("image2 done")
print("========")

plt.show()
print("end program")
