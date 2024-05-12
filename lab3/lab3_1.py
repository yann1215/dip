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


# function: plot gray image
def show_image(img_name, img):
    plt.imshow(img, cmap='gray')
    plt.title(img_name, fontsize=10)
    plt.xticks([])
    plt.yticks([])


# function: check if an image is type RGB; change type to gray
def check_rgb2gray(img):
    if len(img.shape) == 3:  # if image is type RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


# load original image
image_o = np.load("lab1.npy")
# check image type
image_o = check_rgb2gray(image_o)
# apply log transform
image_l = log_transform(image_o, 1)
# plot
plt.subplot(2, 5, 1)
show_image("original image", image_o)
plt.subplot(2, 5, 6)
show_image("log transform image", image_l)

# Step 1
# multiply (-1)^(x+y) and centralize the spectrum
image_o1 = np.zeros_like(image_o)
image_l1 = np.zeros_like(image_l)
for i in range(image_o.shape[0]):
    for j in range(image_o.shape[1]):
        image_o1[i, j] = pow(-1, i + j) * image_o[i, j]
        image_l1[i, j] = pow(-1, i + j) * image_l[i, j]
# plot step 1
plt.subplot(2, 5, 2)
show_image("(1) original image multiplied by (-1)$^{x+y}$", image_o1)
plt.subplot(2, 5, 7)
show_image("(1) log transform image multiplied by (-1)$^{x+y}$", image_l1)

# Step 2: 2-D Fourier Transform
image_o2 = np.fft.fft2(image_o1)
image_l2 = np.fft.fft2(image_l1)
# plot step 2
# the value of a pixel after Fourier Transform is a complex number
# in order to show the image with plt.plot(), abs() is necessary here
plt.subplot(2, 5, 3)
show_image("(2) original image after FT", abs(image_o2))
plt.subplot(2, 5, 8)
show_image("(2) log transform image after FT", abs(image_l2))

# Step 3: Inverse Fourier Transform
image_o3 = np.fft.ifft2(image_o2)
image_l3 = np.fft.ifft2(image_l2)
# plot step 3
plt.subplot(2, 5, 4)
show_image("(3) original image after iFT", abs(image_o3))
plt.subplot(2, 5, 9)
show_image("(3) log transform image after iFT", abs(image_l3))

# Step 4
image_o4 = np.real(image_o3)
image_l4 = np.real(image_l3)
for i in range(image_o4.shape[0]):
    for j in range(image_o4.shape[1]):
        image_o4[i, j] = pow(-1, i + j) * image_o4[i, j]
        image_l4[i, j] = pow(-1, i + j) * image_l4[i, j]
# plot step 4
plt.subplot(2, 5, 5)
show_image("(4) original image", image_o4)
plt.subplot(2, 5, 10)
show_image("(4) log transform image", image_l4)

plt.tight_layout()
plt.show()
