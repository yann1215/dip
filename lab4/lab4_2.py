import numpy as np
import matplotlib.pyplot as plt
import lab_basic as lb
import cv2
# import goto
# from dominate.tags import label
# from goto import with_goto


# @with_goto
def add_ps_noise(img, pn, ps, pp):
    # pn: noise% in image
    # ps: salt% in noise
    # pp: pepper% in noise

    img_size = np.size(img)

    # get salt noise
    salt_num = int(img_size * pn * ps)
    if salt_num != 0:
        salt_pos = np.zeros(img_size)
        salt_pos[0:salt_num - 1] = 1
        np.random.shuffle(salt_pos)
        salt_pos = salt_pos.reshape(img.shape)

        # add salt noise
        img_max = max(img.flatten())
        img = img + img_max * salt_pos
        img = np.where(img > img_max, img_max, img)

    # get pepper noise
    # label.pepper
    pepper_num = int(img_size * pn * pp)
    if pepper_num != 0:
        pepper_pos = np.ones(img_size)
        pepper_pos[0:pepper_num-1] = 0
        np.random.shuffle(pepper_pos)
        pepper_pos = pepper_pos.reshape(img.shape)

        # add pepper noise
        img = np.multiply(img, pepper_pos)

    return img


# function: Harmonic mean filter
def hmf(img):
    m, n = 3, 3
    h_pad = int((m - 1) / 2)
    w_pad = int((n - 1) / 2)

    img_out = img.copy()
    img_pad = np.pad(img, ((h_pad, m - h_pad - 1), (w_pad, n - w_pad - 1)), mode="edge")

    for i in range(h_pad, img.shape[0] + h_pad):
        for j in range(w_pad, img.shape[1] + w_pad):
            k = np.sum(1.0 / (img_pad[i - h_pad:i + m - h_pad, j - w_pad:j + n - w_pad] + 1e-8))  # avoid divided by 0
            img_out[i - h_pad][j - w_pad] = m * n / k

    return img_out


# function: adaptive median filter
def amf(img, max_size):
    origen = 3  # original kernel size
    board = origen // 2
    # max_board = max_size//2
    copy = cv2.copyMakeBorder(img, *[board] * 4, borderType=cv2.BORDER_DEFAULT)
    img_out = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            def sub_func(src, size):
                kernel = src[i:i + size, j:j + size]
                # print(kernel)
                z_med = np.median(kernel)
                z_max = np.max(kernel)
                z_min = np.min(kernel)
                if z_min < z_med < z_max:  # layer A
                    if z_min < img[i][j] < z_max:  # layer B
                        return img[i][j]
                    else:
                        return z_med
                else:
                    next_size = cv2.copyMakeBorder(src, *[1] * 4, borderType=cv2.BORDER_DEFAULT)  # enlarge
                    size = size + 2
                    if size <= max_size:
                        return sub_func(next_size, size)  # repeat layer A
                    else:
                        return z_med

            img_out[i][j] = sub_func(copy, origen)
    return img_out


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

    return img


image0 = np.load("lab1.npy")
image0 = lb.check_rgb2gray(image0)
# plt.subplot(1, 4, 1)
# lb.show_image("original image", image0)
print("image0 loaded")

image1 = add_ps_noise(image0, 0.4, 0, 1)
plt.subplot(1, 4, 1)
lb.show_image("image with noise", image1)
print("image1 loaded")

image2 = amf(image1, 7)
plt.subplot(1, 4, 2)
lb.show_image("adaptive median filter", image2)
print("image2 loaded")

image3 = hmf(image1)
plt.subplot(1, 4, 3)
lb.show_image("Harmonic mean filter", image3)
print("image3 loaded")

image4 = median_filter(image1)
plt.subplot(1, 4, 4)
lb.show_image("median filter", image4)
print("image4 loaded")

plt.tight_layout()
plt.show()
