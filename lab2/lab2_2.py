import cv2
import numpy as np
import matplotlib.pyplot as plt


# define gray scale
gs = 256


# function: get normalized histogram value
def get_hist(img):
    img = img.astype(np.int64)
    img = img.flatten()
    img_hist = np.bincount(img)
    # img_hist.resize(gs, 1)  # meet the requirement of plt.stem
    img_hist = img_hist / img.shape[0]  # calculate probability
    return img_hist


# function: plot normalized histogram
def plot_hist(img):
    img_hist = get_hist(img)
    plt.stem(img_hist, markerfmt="C0")


# function: plot normalized histogram using plt function
def function_plot_hist(img):
    plt.hist(img, bins=64, facecolor="b")  # runs very slow
    plt.xlim(0, gs)


# function: histogram equalization
def hist_equalization(img):
    img_hist = get_hist(img)
    s = np.cumsum(img_hist)
    s = (gs - 1) * (s / s[-1])
    x = np.arange(0, gs, 1)
    img1 = np.interp(img.flatten(), x, s)
    img1 = img1.reshape(img.shape)
    img1 = img1.astype(np.int64)
    return img1


def show_image(img_name, img):
    plt.imshow(img, cmap='gray')
    plt.title(img_name, fontsize=10)
    plt.xticks([])
    plt.yticks([])


# start project
# load and initialize image
image0 = np.load("lab2_1.npy")
image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY) * (gs - 1)  # maximum value change from 1 to (gs -1)

# histogram equalization
image1 = hist_equalization(image0)

# plot original image
plt.subplot2grid((3, 2), (0, 0), rowspan=2)
show_image("original image", image0)
# plot histogram of original image
plt.subplot(3, 2, 5)
plot_hist(image0)
plt.title("histogram of original image", fontsize=10)

# # plot processed image
plt.subplot2grid((3, 2), (0, 1), rowspan=2)
show_image("image after histogram equalization", image1)
# # plot histogram of original image
plt.subplot(3, 2, 6)
plot_hist(image1)
plt.title("histogram of processed image", fontsize=10)

plt.tight_layout()
plt.show()
