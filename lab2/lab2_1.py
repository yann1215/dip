import cv2
import numpy as np
import matplotlib.pyplot as plt


# function: image log transform
def log_transform(img, c):
    # log transform: s = c * log(1 + r)
    # img: image
    # c: log transform parameter

    img = c * np.log(img + 1)
    return img


# function: image power transform
def power_transform(img, c, gamma):
    # power transform: s = c * r ^ gamma
    # img: image
    # c, gamma: power transform parameters

    img = c * np.power(img, gamma)
    return img


def show_image(img_name, img):
    plt.imshow(img, cmap='gray')
    plt.title(img_name, fontsize=10)
    plt.xticks([])
    plt.yticks([])


# start project
# load and initialize image
image0 = np.load("lab2_1.npy")
image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)

image1 = log_transform(image0, 1)
image2 = power_transform(image0, 1, 0.1)

# plot original image
plt.subplot(2, 3, 1)
show_image("original image", image0)
plt.subplot(2, 3, 4)
plt.hist(image0.flatten(), bins=256, color="b")

# plot log transform image
plt.subplot(2, 3, 2)
show_image("log transform image", image1)
plt.subplot(2, 3, 5)
plt.hist(image1.flatten(), bins=256, color="b")

# plot power transform image
plt.subplot(2, 3, 3)
show_image("power transform image", image2)
plt.subplot(2, 3, 6)
plt.hist(image2.flatten(), bins=256, color="b")

plt.tight_layout()
plt.show()
