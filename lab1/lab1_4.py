import cv2
import numpy as np
import matplotlib.pyplot as plt

# set threshold
thres = 200


def replace_color(img, src_clr, dst_clr):
    # src_clr:	color (r,g,b)
    # dst_clr:  target color (r,g,b)

    img_arr = np.asarray(img, dtype=np.double)

    r_img = img_arr[:, :, 0].copy()
    g_img = img_arr[:, :, 1].copy()
    b_img = img_arr[:, :, 2].copy()

    img = r_img * 256 * 256 + g_img * 256 + b_img
    src_color = src_clr[0] * 256 * 256 + src_clr[1] * 256 + src_clr[2]

    r_img[img == src_color] = dst_clr[0]
    g_img[img == src_color] = dst_clr[1]
    b_img[img == src_color] = dst_clr[2]

    dst_img = np.array([r_img, g_img, b_img], dtype=np.uint8)
    dst_img = dst_img.transpose(1, 2, 0)

    return dst_img


# load original image
image_origin = np.load("lab1.npy", 0)

plt.subplot(2, 2, 1)
plt.imshow(image_origin, cmap='gray')
plt.title("original image", fontsize=12)
plt.xticks([])
plt.yticks([])

# get threshold image: mask
ret1, image_mask = cv2.threshold(image_origin, int(thres), 255, cv2.THRESH_BINARY)
cv2.imwrite(f"./saved_image/threshold={thres}.png", image_mask)

plt.subplot(2, 2, 2)
plt.imshow(image_mask, cmap='gray')
plt.title("mask image(before color)", fontsize=12)
plt.xticks([])
plt.yticks([])

# get threshold image: background image
ret2, image_back = cv2.threshold(image_origin, int(thres), 255, cv2.THRESH_BINARY_INV)
image_back = cv2.bitwise_and(image_origin, image_back)
cv2.imwrite(f"./saved_image/background_image(threshold={thres}).png", image_back)

plt.subplot(2, 2, 3)
plt.imshow(image_back, cmap='gray')
plt.title("background image", fontsize=12)
plt.xticks([])
plt.yticks([])

# get red mask
image_mask = cv2.cvtColor(image_mask, cv2.COLOR_GRAY2RGB)
image_mask = replace_color(image_mask, (255, 255, 255), (255, 0, 0)).astype(np.uint8)  # image type required to uniform

# mask the original image
image_back = cv2.cvtColor(image_back, cv2.COLOR_GRAY2RGB).astype(np.uint8)
image_out = cv2.add(image_back, image_mask)
cv2.imwrite(f"./saved_image/masked_image(threshold={thres}).png", image_out)

plt.subplot(2, 2, 4)
plt.imshow(image_out)
plt.title(f"masked image(threshold = {thres})", fontsize=12)
plt.xticks([])
plt.yticks([])

plt.show()
