import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Keywords: Laplacian enhancement


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


# function: 8 neighbour Laplacian filter
def laplacian_filter(img):
    # get 8 neighbour Laplacian kernel
    lk = np.full((3, 3), -1)
    lk[1, 1] = 8
    # Laplacian filter
    img = signal.convolve2d(img, lk, "same")
    return abs(img)


# function: Laplacian enhancement
def laplacian_enhancement(img, c):
    img = img + c * laplacian_filter(img)
    return img


image0 = np.load("lab2.npy")
image0 = check_rgb2gray(image0)
plt.subplot(1, 3, 1)
show_image("original image", image0)
print("image0 loaded")
print("========")

image1 = laplacian_filter(image0)
plt.subplot(1, 3, 2)
show_image("Laplacian filtered image", image1)
print("image1 done")
print("========")

image2 = laplacian_enhancement(image0, 1)
plt.subplot(1, 3, 3)
show_image("Laplacian enhanced image", image2)
print("image2 done")
print("========")

plt.show()
print("end program")
