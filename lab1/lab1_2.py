import cv2
import numpy as np
import matplotlib.pyplot as plt


class IMG:
    def __init__(self):
        self.image = []
        self.name = ""


image256 = IMG()
image256.image = np.load("lab1/lab1.npy")
image256.name = "grey level 256"
cv2.imwrite("saved_image/"+image256.name+".png", image256.image)


def reduce_intensity_levels(image_in, level):
    image_out = cv2.copyTo(image_in, None)
    for x in range(image_out.shape[0]):
        for y in range(image_out.shape[1]):
            grey1 = image_out[x, y]
            grey2 = int(level * grey1 / 255 + 0.5) * (255 / level)
            image_out[x, y] = grey2
    return image_out


def plot_four_images(img1, img2, img3, img4):
    plt.subplot(2, 2, 1)
    plt.imshow(img1.image, cmap='gray')
    plt.title(img1.name, fontsize=8)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 2)
    plt.imshow(img2.image, cmap='gray')
    plt.title(img2.name, fontsize=8)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 3)
    plt.imshow(img3.image, cmap='gray')
    plt.title(img3.name, fontsize=8)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 4)
    plt.imshow(img4.image, cmap='gray')
    plt.title(img4.name, fontsize=8)
    plt.xticks([])
    plt.yticks([])

    plt.show()


image64 = IMG()
image64.image = reduce_intensity_levels(image256.image, 64)
image64.name = "grey level 64"
cv2.imwrite("saved_image/"+image64.name+".png", image64.image)

image16 = IMG()
image16.image = reduce_intensity_levels(image256.image, 16)
image16.name = "grey level 16"
cv2.imwrite("saved_image/"+image16.name+".png", image16.image)

image4 = IMG()
image4.image = reduce_intensity_levels(image256.image, 4)
image4.name = "grey level 4"
cv2.imwrite("saved_image/"+image4.name+".png", image4.image)

plot_four_images(image256, image64, image16, image4)
