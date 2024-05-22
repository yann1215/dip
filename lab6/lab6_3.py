import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from queue import Queue
from sklearn.cluster import KMeans
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import lab_basic as lb


def region_growing(image, seed, threshold):
    segmented = np.zeros_like(image)

    # use a queue to store pixels to be grown
    q = Queue()
    q.put(seed)

    # growing condition function to determine if two pixels are similar
    def condition(p1, p2):
        return abs(int(p1) - int(p2)) < threshold

    # start growing
    while not q.empty():
        current_point = q.get()
        x, y = current_point

        # mark the current pixel as visited
        segmented[x, y] = 255

        # check the 8 neighboring pixels of the current pixel
        for i in range(-1, 2):
            for j in range(-1, 2):
                # make sure not to go out of image bounds
                if (0 <= x + i < image.shape[0]) and (0 <= y + j < image.shape[1]):
                    # check if the pixel is unvisited and meets the growing condition
                    if segmented[x + i, y + j] == 0 and condition(image[x, y], image[x + i, y + j]):
                        # add the pixel to the queue if it meets the condition
                        q.put((x + i, y + j))
                        # mark the current pixel as visited
                        segmented[x + i, y + j] = 255

    return segmented


def kmeans_segmentation(image, num_clusters):
    # Reshape the image into a 2D array for clustering
    reshaped_image = image.reshape((-1, 1))
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(reshaped_image)
    # Get the labels assigned to each pixel
    labels = kmeans.labels_
    # Reshape the labels back to the original image shape
    segmented_image = labels.reshape(image.shape)

    return segmented_image


def global_linear_transformation(img):
    maxV = img.max()
    minV = img.min()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = ((img[i, j] - minV) * 255) / (maxV - minV)
    return img


def watershed_segmentation(image):
    # Convert image to 8-bit grayscale
    gray = cv.normalize(image, None, 0, 255, cv.NORM_MINMAX).astype('uint8')

    # Apply a binary threshold to the image
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Compute the distance transform
    distance = ndi.distance_transform_edt(binary)

    # Find local maxima in the distance transform
    local_maxi = peak_local_max(distance, labels=binary)

    # Perform connected component analysis on the local maxima
    markers, _ = ndi.label(local_maxi)

    # Ensure markers have the same shape as the distance array
    markers = ndi.zoom(markers, np.array(distance.shape) / np.array(markers.shape))

    # Apply the watershed algorithm
    labels = 1 - watershed(-distance, markers, mask=binary)  # mark brain with label 1

    return labels


image = np.load("lab6_t2w.npy")
image_blurred = cv.blur(image, (3, 3))

# ---------- Region Growing ----------
seed_point = (110, 75)  # (y, x)
threshold_value = 32
segmented_image = region_growing(image_blurred, seed_point, threshold_value)
print("region growing:", np.sum(segmented_image[segmented_image == 255])/255)

plt.figure(1)
plt.subplot(1, 2, 1)
lb.show_image("original image", image)
plt.subplot(1, 2, 2)
lb.show_image("segmented image(region growth)", segmented_image)
plt.show()

# ---------- K-means Cluster ----------
num_clusters = 4
k_segmented_image = kmeans_segmentation(image_blurred, num_clusters)
print("k-means cluster:", np.sum(k_segmented_image[k_segmented_image == 3])/3)

plt.figure(2)
plt.subplot(1, 3, 1)
lb.show_image("original image", image)
plt.subplot(1, 3, 2)
lb.show_image("segmented image(k-means cluster)", k_segmented_image)

k_segmented_image[k_segmented_image != 3] = 0
plt.subplot(1, 3, 3)
lb.show_image("lesion part(k-means cluster)", k_segmented_image)

plt.show()

# ---------- Watershed ----------
image_w = np.copy(image_blurred)
image_w[image_w < 1100] = 0
w_segmented_image = watershed_segmentation(image_w)
print("watershed:", np.sum(w_segmented_image[w_segmented_image == 1]))

plt.figure(3)
plt.subplot(1, 2, 1)
lb.show_image("original image", image_blurred)
plt.subplot(1, 2, 2)
lb.show_image("segmented image(watershed)", w_segmented_image)
plt.show()
