import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import lab_basic as lb


def global_linear_transformation(img):
    maxV = img.max()
    minV = img.min()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = ((img[i, j] - minV) * 255) / (maxV - minV)
    return img


def calculate_global_threshold(img):
    img_threshold = int(np.mean(img[img > min_intensity]))
    last_threshold = 0

    error_img = 1
    error_set = 0.01

    while error_img > error_set:
        c1_mean = np.mean(img[(img > img_threshold) & (img > min_intensity)])
        c2_mean = np.mean(img[(img <= img_threshold) & (img > min_intensity)])
        # print(c1_mean, c2_mean)
        img_threshold = int((c1_mean + c2_mean) / 2)

        error_img = (img_threshold - last_threshold) / img_threshold
        last_threshold = img_threshold

    return img_threshold


def get_hist(img):
    img = img.astype(np.int64)
    img = img.flatten()
    img_hist = np.bincount(img)
    # img_hist.resize(gs, 1)  # meet the requirement of plt.stem
    img_hist = img_hist / img.shape[0]  # calculate probability
    return img_hist


def otsu_threshold(img):
    img_hist = get_hist(img[img > min_intensity])
    img_hist_max = len(img_hist)
    img_sum = sum(img_hist)

    img_threshold = 0
    threshold_count = 1

    current_var = 0
    max_var = 0

    for t in range(img_hist_max - 1):
        if t <= min_intensity:
            continue
        p1 = sum(img_hist[t:img_hist_max-1]) / img_sum
        p2 = sum(img_hist[0:t-1]) / img_sum
        c1_mean = np.mean(img[(img > t) & (img > min_intensity)])
        c2_mean = np.mean(img[(img <= t) & (img > min_intensity)])

        current_var = p1 * p2 * (c1_mean - c2_mean) ** 2

        if current_var > max_var:
            max_var = current_var
            img_threshold = t
            threshold_count = 1
        elif current_var == max_var:
            img_threshold += t
            threshold_count += 1

    img_threshold = int(img_threshold / threshold_count)
    return img_threshold


# ---------- Load Original Image ----------
t1 = np.load("lab6_t1gd.npy")
t1 = lb.check_rgb2gray(t1)
t1 = global_linear_transformation(t1)

t2 = np.load("lab6_t2w.npy")
t2 = lb.check_rgb2gray(t2)
t2 = global_linear_transformation(t2)

min_intensity = 100

# ---------- Process Image: Method 1, Global Thresholding ----------
# calculate global threshold
t1_global_threshold = calculate_global_threshold(t1)
print(f'Global Threshold for T1-Gd: {t1_global_threshold}')
t2_global_threshold = calculate_global_threshold(t2)
print(f'Global Threshold for T2w: {t2_global_threshold}')

# apply global threshold
ret1, t1_global = cv.threshold(t1, t1_global_threshold, 255, cv.THRESH_BINARY)
ret2, t2_global = cv.threshold(t2, t2_global_threshold, 255, cv.THRESH_BINARY)

# ---------- Process Image: Method 2, Otsu Thresholding ----------
t1_otsu_threshold = otsu_threshold(t1)
print(f'Otsu Threshold for T1-Gd: {t1_otsu_threshold}')
t2_otsu_threshold = otsu_threshold(t2)
print(f'Otsu Threshold for T1-Gd: {t2_otsu_threshold}')

# apply Otsu threshold
ret1_otsu, t1_otsu = cv.threshold(t1, t1_otsu_threshold, 255, cv.THRESH_BINARY)
ret2_otsu, t2_otsu = cv.threshold(t2, t2_otsu_threshold, 255, cv.THRESH_BINARY)

# ---------- Plot Image----------
# plot histogram
plt.figure(1)

plt.subplot(1, 2, 1)
plt.hist(t1[t1 > min_intensity].flatten(), bins=100)
plt.subplot(1, 2, 2)
plt.hist(t2[t2 > min_intensity].flatten(), bins=100)
plt.show()

# plot image
plt.figure(2)

plt.subplot(2, 3, 1)
lb.show_image("T1-Gd original image", t1)
plt.subplot(2, 3, 2)
lb.show_image("T1-Gd after global thresholding", t1_global)
plt.subplot(2, 3, 3)
lb.show_image("T1-Gd after Otsu thresholding", t1_otsu)

plt.subplot(2, 3, 4)
lb.show_image("T2w original image", t2)
plt.subplot(2, 3, 5)
lb.show_image("T2w after global thresholding", t2_global)
plt.subplot(2, 3, 6)
lb.show_image("T2w after Otsu thresholding", t2_otsu)

plt.show()
