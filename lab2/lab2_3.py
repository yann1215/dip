import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def img_trans(img, ax, pos, c):
    # ax: axis the image move along
    # pos: r -> right, l -> left, u -> up, d -> down
    # c: the value of translation

    if ax == 'x':
        if pos == 'l':
            c = -c
        elif pos != 'r':
            c = 0
        img1 = np.roll(img, c, axis=1)
    elif ax == 'y':
        if pos == 'u':
            c = -c
        elif pos != 'd':
            c = 0
        img1 = np.roll(img, c, axis=0)
    else:
        img1 = img

    return img1


def img_rescale(img, r_scale, l_scale):
    r_new = int(img.shape[0] * r_scale)
    l_new = int(img.shape[1] * l_scale)
    img1 = np.zeros((r_new, l_new))

    for ri in range(r_new):
        for li in range(l_new):
            ri_old = int(ri / r_scale)
            li_old = int(li / l_scale)
            img1[ri, li] = img[ri_old, li_old]
    return img1


def img_flip(img, flip_axis):

    r_max = img.shape[0]
    l_max = img.shape[1]
    img1 = np.zeros((r_max, l_max))

    if flip_axis == 'x':
        for ri in range(r_max):
            img1[ri, :] = img[r_max - ri - 1, :]
    elif flip_axis == 'y':
        for li in range(l_max):
            img1[:, li] = img[:, l_max - li - 1]

    return img1


def img_rotate(img, flag, deg):
    # flag: 0-> clockwise, 1 -> counter-clockwise
    # deg: degree (not radian)

    if flag == 0:
        deg = -deg

    h, w = img.shape[:2]
    rotate_center = (w / 2, h / 2)

    # calculate rotation matrix
    M = cv2.getRotationMatrix2D(rotate_center, deg, 1.0)

    # calculate new boundary
    new_w = int(h * np.abs(M[0, 1]) + w * np.abs(M[0, 0]))
    new_h = int(h * np.abs(M[0, 0]) + w * np.abs(M[0, 1]))

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    img1 = cv2.warpAffine(img, M, (new_w, new_h))

    return img1


def show_image(img_name, img):
    plt.imshow(img, cmap='gray')
    plt.title(img_name, fontsize=10)
    # plt.xticks([])
    # plt.yticks([])


# start project
# load and initialize image
image0 = np.load("lab2_1.npy")
image0 = cv2.cvtColor(image0, cv2.COLOR_RGB2GRAY)

# function 1: translate
image1 = img_trans(image0, 'y', 'u', 400)
# function 2: rescale
image2 = img_rescale(image0, 0.5, 0.8)
# function 3: flip
image3 = img_flip(image0, 'x')
# function 4: rotate
image4 = img_rotate(image0, 0, 45)

# plot all images
plt.subplot(2, 2, 1)
show_image("translated image", image1)
plt.subplot(2, 2, 2)
show_image("rescaled image", image2)
plt.subplot(2, 2, 3)
show_image("flipped image", image3)
plt.subplot(2, 2, 4)
show_image("rotated image", image4)

plt.show()
