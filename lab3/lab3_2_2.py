import cv2
import numpy as np
import matplotlib.pyplot as plt

# Keywords: median filter, dalt-and-pepper noise, brain MRI


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


# function: median filter
def median_filter(img):

    # get kernel value
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # reset kernel value
            k = np.zeros((3, 3))

            # check if pixel is on the edge of the image and get kernel value
            if i == 0:
                if j == 0:
                    k[1, 1] = img[i, j]
                    k[1, 2] = img[i, j + 1]
                    k[2, 1] = img[i + 1, j]
                    k[2, 2] = img[i + 1, j + 1]
                elif j == (img.shape[1] - 1):
                    k[1, 0] = img[i, j - 1]
                    k[1, 1] = img[i, j]
                    k[2, 0] = img[i + 1, j - 1]
                    k[2, 1] = img[i + 1, j]
                else:
                    k[1, 0] = img[i, j - 1]
                    k[1, 1] = img[i, j]
                    k[1, 2] = img[i, j + 1]
                    k[2, 0] = img[i + 1, j - 1]
                    k[2, 1] = img[i + 1, j]
                    k[2, 2] = img[i + 1, j + 1]
            elif i == (img.shape[0] - 1):
                if j == 0:
                    k[0, 1] = img[i - 1, j]
                    k[0, 2] = img[i - 1, j + 1]
                    k[1, 1] = img[i, j]
                    k[1, 2] = img[i, j + 1]
                elif j == (img.shape[1] - 1):
                    k[0, 0] = img[i - 1, j - 1]
                    k[0, 1] = img[i - 1, j]
                    k[1, 0] = img[i, j - 1]
                    k[1, 1] = img[i, j]
                else:
                    k[0, 0] = img[i - 1, j - 1]
                    k[0, 1] = img[i - 1, j]
                    k[0, 2] = img[i - 1, j + 1]
                    k[1, 0] = img[i, j - 1]
                    k[1, 1] = img[i, j]
                    k[1, 2] = img[i, j + 1]
            else:
                if j == 0:
                    k[0, 1] = img[i - 1, j]
                    k[0, 2] = img[i - 1, j + 1]
                    k[1, 1] = img[i, j]
                    k[1, 2] = img[i, j + 1]
                    k[2, 1] = img[i + 1, j]
                    k[2, 2] = img[i + 1, j + 1]
                elif j == (img.shape[1] - 1):
                    k[0, 0] = img[i - 1, j - 1]
                    k[0, 1] = img[i - 1, j]
                    k[1, 0] = img[i, j - 1]
                    k[1, 1] = img[i, j]
                    k[2, 0] = img[i + 1, j - 1]
                    k[2, 1] = img[i + 1, j]
                else:
                    k[0, 0] = img[i - 1, j - 1]
                    k[0, 1] = img[i - 1, j]
                    k[0, 2] = img[i - 1, j + 1]
                    k[1, 0] = img[i, j - 1]
                    k[1, 1] = img[i, j]
                    k[1, 2] = img[i, j + 1]
                    k[2, 0] = img[i + 1, j - 1]
                    k[2, 1] = img[i + 1, j]
                    k[2, 2] = img[i + 1, j + 1]

            img[i, j] = np.median(k)

        print(f"line {i} finished")

    return img


# original image
image0 = np.load("lab1.npy")
image0 = check_rgb2gray(image0)
plt.subplot(2, 2, 1)
show_image("original image", image0)

# image with noise
image1 = np.load("noisyMRI.npy")
image1 = check_rgb2gray(image1)
plt.subplot(2, 2, 3)
show_image("image with noise", image1)

# original image after median filter
image0f = median_filter(image0)
plt.subplot(2, 2, 2)
show_image("original image after median filter", image0f)

# image with noise after median filter
image1f = median_filter(image1)
plt.subplot(2, 2, 4)
show_image("image with noise after median filter", image1f)

plt.tight_layout()
plt.show()
