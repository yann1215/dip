import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import lab_basic as lb


def sobel_filter(img):
    sobel_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_x = cv.convertScaleAbs(sobel_x)
    sobel_y = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.convertScaleAbs(sobel_y)
    img_sobel = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    return img_sobel


def smooth_filter(img):
    # mean filter
    kernel_size = (3, 3)
    img_smooth = cv.blur(img, kernel_size)

    # median filter
    # kernel_size = 3
    # img_smooth = cv.medianBlur(img, kernel_size)

    # gaussian filter
    # kernel_size = (3, 3)
    # sigma_gaussian = 0
    # img_smooth = cv.GaussianBlur(img, kernel_size, sigma_gaussian)
    return img_smooth


# ---------- Load Original Image ----------
mask_threshold = 254  # max value of sobel filtered t1 and t2 is 255
smooth_mask_threshold = 240

t1 = np.load("lab6_t1gd.npy")
t1 = lb.check_rgb2gray(t1)
t2 = np.load("lab6_t2w.npy")
t2 = lb.check_rgb2gray(t2)

# ---------- Process Image: Method 1 ----------
# sobel filter
t1_sobel = sobel_filter(t1)
# print(max(t1_sobel.flatten()))
t2_sobel = sobel_filter(t2)
# print(max(t2_sobel.flatten()))

# binary mask
ret1, t1_mask = cv.threshold(t1_sobel, mask_threshold, 255, cv.THRESH_BINARY)
ret2, t2_mask = cv.threshold(t2_sobel, mask_threshold, 255, cv.THRESH_BINARY)

# ---------- Plot Image: Method 1 ----------
# plt.figure(1)
# plt.subplot(1, 2, 1)
# lb.show_image("T1-Gd original image", t1)
# plt.subplot(1, 2, 2)
# lb.show_image("T2w original image", t2)

# plt.figure(2)
# plt.subplot(1, 2, 1)
# lb.show_image("T1-Gd sobel filtered image", t1_sobel)
# plt.subplot(1, 2, 2)
# lb.show_image("T2w sobel filtered image", t2_sobel)

# plt.figure(3)
# plt.subplot(1, 2, 1)
# plt.hist(t1_sobel.flatten(), bins=32)
# plt.subplot(1, 2, 2)
# plt.hist(t2_sobel.flatten(), bins=32)

plt.figure(4)
plt.subplot(1, 2, 1)
lb.show_image("T1-Gd mask", t1_mask)
plt.subplot(1, 2, 2)
lb.show_image("T2w mask", t2_mask)

plt.show()
plt.tight_layout()

# ---------- Process Image: Method 2 ----------
# smooth filter
t1_smooth = smooth_filter(t1)
t2_smooth = smooth_filter(t2)

# sobel filter
t1_smooth_sobel = sobel_filter(t1_smooth)
t2_smooth_sobel = sobel_filter(t2_smooth)

# binary mask
smooth_ret1, t1_smooth_mask = cv.threshold(t1_smooth_sobel, smooth_mask_threshold, 255, cv.THRESH_BINARY)
smooth_ret2, t2_smooth_mask = cv.threshold(t2_smooth_sobel, smooth_mask_threshold, 255, cv.THRESH_BINARY)

# ---------- Plot Image: Method 2 ----------
# plt.figure(1)
# plt.subplot(1, 2, 1)
# lb.show_image("T1-Gd mean image", t1_smooth)
# plt.subplot(1, 2, 2)
# lb.show_image("T2w mean image", t2_smooth)
#
# plt.figure(2)
# plt.subplot(1, 2, 1)
# lb.show_image("T1-Gd sobel filtered image", t1_smooth_sobel)
# plt.subplot(1, 2, 2)
# lb.show_image("T2w sobel filtered image", t2_smooth_sobel)
#
# plt.figure(3)
# plt.subplot(1, 2, 1)
# plt.hist(t1_smooth_sobel.flatten(), bins=32)
# plt.subplot(1, 2, 2)
# plt.hist(t2_smooth_sobel.flatten(), bins=32)

plt.figure(4)
plt.subplot(1, 2, 1)
lb.show_image("T1-Gd mask", t1_smooth_mask)
plt.subplot(1, 2, 2)
lb.show_image("T2w mask", t2_smooth_mask)

plt.show()
plt.tight_layout()
