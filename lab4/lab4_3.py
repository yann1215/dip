import numpy as np
import matplotlib.pyplot as plt
import lab_basic as lb
# import cv2
import math
from scipy.signal import wiener


def distance(point1, point2):
    d = math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return d


# function: gaussian lowpass filter in frequency domain
def gaussianLP(D0, img):
    imgShape = img.shape
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = math.exp(((-distance((y, x), center)**2)/(2*(D0**2))))
    return base


def add_gaussian_noise(img, m, s):
    gauss = np.random.normal(m, s, img.shape)
    img = img + gauss
    return img


def image_fft(img):
    img_spec = np.fft.fft2(img)
    img_spec = np.fft.fftshift(img_spec)
    return img_spec


def image_ifft(img_spec):
    img = np.fft.ifftshift(img_spec)
    img = np.fft.ifft2(img)
    img = np.real(img)
    return img


def limited_inverse_filter(Df, imgMask):
    imgShape = imgMask.shape
    limMask = np.copy(imgMask)
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            if (distance((y, x), center) >= Df):
                limMask[y, x] = 0
    return limMask


# definition
D0 = 50     # gaussian lowpass filter radius in frequency domain
Df = 30     # limited inverse filter radius

# load original image
image_orig = np.load("lab1.npy")
image_orig = lb.check_rgb2gray(image_orig)
plt.subplot(2, 2, 1)
lb.show_image("original image", image_orig)

# Fourier transform
# image_orig = np.float32(image_orig)
image_spec = image_fft(image_orig)

# gaussian filter in frequency domain
gaussianMask = gaussianLP(D0, image_spec)
plt.subplot(2, 2, 2)
lb.show_image("gaussian filter mask in frequency domain", np.real(gaussianMask))

image_lp_spec = image_spec * gaussianMask
image_lp = image_ifft(image_lp_spec)
image_lp = np.real(image_lp)
plt.subplot(2, 2, 3)
lb.show_image("gaussian filtered image in spatial domain", image_lp)

# add gaussian noise
image_noise = add_gaussian_noise(image_lp, 0, 9)
plt.subplot(2, 2, 4)
lb.show_image("image with noise in spatial domain", image_noise)

plt.show()

plt.subplot(2, 2, 1)
lb.show_image("image with noise in spatial domain", image_noise)

# inverse filter
image_noise_spec = image_fft(image_noise)
image_inv_spec = image_noise_spec / gaussianMask
image_inv = image_ifft(image_inv_spec)
plt.subplot(2, 2, 2)
lb.show_image("inverse filtered image", image_inv)

# limited inverse filter
limMask = limited_inverse_filter(Df, gaussianMask)
image_lim_spec = image_noise_spec / limMask
image_lim = image_ifft(image_lim_spec)
plt.subplot(2, 2, 3)
lb.show_image("limited inverse filtered image", image_inv)

# Wiener filter
image_w = wiener(image_noise)
plt.subplot(2, 2, 4)
lb.show_image("Wiener filtered image", image_w)

plt.show()
