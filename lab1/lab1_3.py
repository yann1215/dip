import cv2
import numpy as np
import matplotlib.pyplot as plt


class IMG:
    def __init__(self):
        self.image = []
        self.name = ""
        self.size = 1


line = 50  # plot details of column 50
n = 4  # amplify & reduce rate

image0 = IMG()
image0.image = np.load("lab1.npy")
image0.name = "original image (size 1)"

image1 = IMG()
image1.size = n
image1.image = cv2.resize(image0.image, dsize=None, fx=image1.size, fy=image1.size)
image1.name = f"image size {image1.size}"

image2 = IMG()
image2.size = 1 / n
image2.image = cv2.resize(image0.image, dsize=None, fx=image2.size, fy=image2.size)
image2.name = f"image size {image2.size}"


def subplot_image(img, i):
    # plot image
    plt.subplot(2, 3, i)
    plt.imshow(img.image, cmap='gray')

    # mark line
    img_line = int(img.size * line)  # int() is required
    line_x = [img_line, img_line]
    line_y = [0, img.image.shape[0]]
    plt.plot(line_x, line_y, color="red", linewidth=1)

    plt.title(img.name, fontsize=8)

    # plot line details
    details = img.image[:, img_line]
    plt.subplot(2, 3, 3+i)
    plt.plot(details, linewidth=1)


def plot_all_images():
    # minimum
    subplot_image(image2, 1)
    # medium
    subplot_image(image0, 2)
    # maximum
    subplot_image(image1, 3)
    plt.savefig("./saved_image/lab1_3.png")
    plt.show()


plot_all_images()
