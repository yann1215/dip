import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import sobel, prewitt, roberts


def morphological_processing(mask):
    # Define a 5x5 kernel for morphological operations
    kernel = np.ones((5, 5), np.uint8)

    # Erosion
    mask_out = cv2.erode(mask, kernel, iterations=1)

    # Dilation
    mask_out = cv2.dilate(mask_out, kernel, iterations=1)

    # Opening (erosion followed by dilation)
    mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_OPEN, kernel)

    # Closing (dilation followed by erosion)
    mask_out = cv2.morphologyEx(mask_out, cv2.MORPH_CLOSE, kernel)

    return mask_out


def edge_detection(image):
    # Apply Sobel operator
    sobel_edges = sobel(image)

    # Apply Prewitt operator
    prewitt_edges = prewitt(image)

    # Apply Roberts operator
    roberts_edges = roberts(image)

    return sobel_edges, prewitt_edges, roberts_edges


def calculate_centroid(mask):
    # Calculate the moments of the mask
    M = cv2.moments(mask)

    # Calculate x, y coordinates of the centroid
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    return (cX, cY)


def calculate_polar_coordinates(edges, centroid):
    cX, cY = centroid
    indices = np.argwhere(edges > 0)

    distances = np.sqrt((indices[:, 1] - cX) ** 2 + (indices[:, 0] - cY) ** 2)
    angles = np.arctan2(indices[:, 0] - cY, indices[:, 1] - cX)

    return distances, angles


def plot_signature(signature, angle):
    sorted_edge = sorted(zip(sobel_angle, sobel_signature))
    sorted_edge = np.array(sorted_edge)
    # print(sorted.shape)
    plt.plot(sorted_edge[:, 0], sorted_edge[:, 1])


# ---------- Load Image ----------
# Paths to the image and mask files
image_path = "tumor.npy"
mask_path = "mask.npy"

# Load the image and mask from npy files
image = np.load(image_path)
mask = np.load(mask_path)

# ---------- Get Edges ----------
# Perform morphological processing on the mask
mask = morphological_processing(mask)

# Perform edge detection on the image
sobel_edges, prewitt_edges, roberts_edges = edge_detection(mask)

# ---------- Get Signature ----------
centroid = calculate_centroid(mask)

# Calculate signatures for the edge-detected images
sobel_signature, sobel_angle = calculate_polar_coordinates(sobel_edges, centroid)
prewitt_signature, prewitt_angle = calculate_polar_coordinates(prewitt_edges, centroid)
roberts_signature, roberts_angle = calculate_polar_coordinates(sobel_edges, centroid)
# sobel_signature = calculate_signature(sobel_edges)
# prewitt_signature = calculate_signature(prewitt_edges)
# roberts_signature = calculate_signature(roberts_edges)

# ---------- Plot Edges ----------
# Visualize the results
plt.figure(figsize=(12, 8))

plt.subplot(231), plt.title("Sobel edges"), plt.imshow(sobel_edges, cmap="gray")
plt.scatter(*centroid, color="red", s=10)
plt.xticks([]), plt.yticks([])

plt.subplot(232), plt.title("Prewitt edges"), plt.imshow(prewitt_edges, cmap="gray")
plt.scatter(*centroid, color="red", s=10)
plt.xticks([]), plt.yticks([])

plt.subplot(233), plt.title("Robert edges"), plt.imshow(roberts_edges, cmap="gray")
plt.scatter(*centroid, color="red", s=10)
plt.xticks([]), plt.yticks([])

# ---------- Plot Signature ----------
plt.subplot(234), plt.title("Sobel edges signature")
# plt.plot(sobel_angle, sobel_signature)
# plt.scatter(sobel_angle, sobel_signature, s=5)
plot_signature(sobel_signature, sobel_angle)

plt.subplot(235), plt.title("Prewitt edges signature")
plot_signature(prewitt_signature, prewitt_angle)

plt.subplot(236), plt.title("Robert edges signature")
plot_signature(roberts_signature, roberts_angle)


plt.tight_layout()
plt.show()
