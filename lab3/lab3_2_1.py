import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal

# Keywords: Gaussian filter, brain MRI
# sigma in Gaussian filter requires to be 1, 3 and 5


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

    print("start Gaussian filter")
    gk = get_gaussian_kernel(s)  # Gaussian kernel
    img = signal.convolve2d(img, gk, "same")
    print("Gaussian filter done")

    return img


# function: get the Gaussian kernel
def get_gaussian_kernel(s):
    # s: sigma in Gaussian filter

    print(f"start calculating Gaussian kernel with sigma {s}")

    a = 6 * s + 1  # size of the kernel
    mid = 3 * s  # position of the kernel center
    k = np.zeros((a, a))
    c = math.exp(- pow(mid, 2) / pow(s, 2))  # center value

    for i in range(-mid, mid + 1):
        for j in range(-mid, mid + 1):
            k[mid + i, mid + j] = math.exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(s, 2))) / c  # pixel value

    k_sum = np.sum(k)
    k = k / k_sum

    print(f"Gaussian kernel with sigma {s} got")

    return k


image0 = np.load("lab1.npy")
image0 = check_rgb2gray(image0)
plt.subplot(2, 2, 1)
show_image("original image", image0)
print("image0 loaded")
print("========")

image1 = gaussian_filter(image0, 1)
plt.subplot(2, 2, 2)
show_image(r"Gaussian filter($\sigma$ = 1)", image1)
print("image1 done")
print("========")

image2 = gaussian_filter(image0, 3)
plt.subplot(2, 2, 3)
show_image(r"Gaussian filter($\sigma$ = 3)", image2)
print("image2 done")
print("========")

image3 = gaussian_filter(image0, 5)
plt.subplot(2, 2, 4)
show_image(r"Gaussian filter($\sigma$ = 5)", image3)
print("image3 done")
print("========")

plt.show()
print("end program")
