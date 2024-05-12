import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

# Keywords: Sobel filter


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


# function: Sobel filter, observe edges in y direction
def sobel_y(img):
    # get Sobel kernel
    sk_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Sobel filter
    img = signal.convolve2d(img, sk_x, "same")

    return img


# function: Sobel filter, observe edges in x direction
def sobel_x(img):
    # get Sobel kernel
    sk_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Sobel filter
    img = signal.convolve2d(img, sk_y, "same")

    return img


# function: 3 * 3 Sobel filter in 2 directions
def sobel_filter(img):
    # get Sobel kernel
    sk_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sk_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # sk = np.sqrt(np.square(sk_x) + np.square(sk_y))
    sk = abs(sk_x) + abs(sk_y)

    # Sobel filter
    img = signal.convolve2d(img, sk, "same")

    return img


image0 = np.load("lab2.npy")
image0 = check_rgb2gray(image0)
plt.subplot(1, 4, 1)
show_image("original image", image0)
print("image0 loaded")
print("========")

image1 = sobel_x(image0)
plt.subplot(1, 4, 2)
show_image("horizontal radiant", abs(image1))
print("image1 done")
print("========")

image2 = sobel_y(image0)
plt.subplot(1, 4, 3)
show_image("vertical radiant", abs(image2))
print("image2 done")
print("========")

image3 = sobel_filter(image0)
plt.subplot(1, 4, 4)
show_image("Sobel filtered image", abs(image3))
print("image3 done")
print("========")

plt.show()
print("end program")
