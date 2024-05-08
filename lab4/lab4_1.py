import numpy as np
import matplotlib.pyplot as plt
import lab_basic as lb


def add_gaussian_noise(img, m, s):
    gauss = np.random.normal(m, s, img.shape)
    img = img + gauss
    return img


image0 = np.load("lab1.npy")
image0 = lb.check_rgb2gray(image0)

m = 0
s = 25
image1 = add_gaussian_noise(image0, 0, 25)

image2 = image1[0:63][0:7]

plt.subplot(2, 3, 1)
lb.show_image("original image", image0)
print("image0 loaded")

plt.subplot(2, 3, 2)
lb.show_image(rf"image with noise($\mu$={m}, $\sigma$={s})", image1)
print("image1 loaded")

# see image in the frequency domain; noise
plt.subplot(2, 3, 3)
lb.show_image("image sample", image2)
print("image2 loaded")

plt.subplot(2, 3, 4)
plt.hist(image0.flatten(), bins=256)
plt.title("histogram of original image")
plt.axis([-50, 300, 0, 15000])
print("histogram of image0 loaded")

plt.subplot(2, 3, 5)
plt.hist(image1.flatten(), bins=256)
plt.title("histogram of image with noise")
plt.axis([-50, 300, 0, 15000])
print("histogram of image1 loaded")

plt.subplot(2, 3, 6)
plt.hist(image2.flatten(), bins=256)
plt.title("histogram of image sample")
# plt.axis([-50, 300, 0, 15000])
print("histogram of image2 loaded")

plt.show()
