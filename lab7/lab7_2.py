import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


def calculate_homogeneity(glcm):
    # print(glcm.shape)
    rows, cols = glcm.shape[:2]
    homogeneity = 0.0
    for i in range(rows):
        for j in range(cols):
            # according to function graycomatrix()
            homogeneity += glcm[i, j, 0, 0] / (1 + abs(i - j) ** 2)
            # according to lecture
            # homogeneity += glcm[i, j, 0, 0] / (1 + abs(i - j))
    return homogeneity


def calculate_contrast(glcm):
    rows, cols = glcm.shape[:2]
    contrast = 0.0
    for i in range(rows):
        for j in range(cols):
            # print(i, j)
            contrast += glcm[i, j, 0, 0] * ((i - j) ** 2)
    return contrast


def calculate_energy(glcm):
    # according to function graycomatrix()
    energy = np.sqrt(np.sum(glcm[:, :, 0, 0] ** 2))
    # according to lecture
    # energy = np.sum(glcm[:, :, 0, 0] ** 2)
    return energy


# ---------- Load image ----------
# Paths to the image and mask files
image_path = "tumor.npy"
mask_path = "mask.npy"

# Load the image and mask from npy files
image = np.load(image_path)
mask = np.load(mask_path)

is_tumor = 0
if is_tumor:
    # process tumor region
    image = np.multiply(image, mask)
    print("====== Processing tumor region ======")
else:
    # process other region
    image = np.multiply(image, (np.max(mask)-mask))
    print("====== Processing other region ======")

# ---------- Rescale the region to 128 levels ----------
image_rescaled = (image * 127 / np.max(image)).astype(np.uint8)
# plt.figure(), plt.imshow(image_rescaled, cmap="gray")
# if is_tumor:
#     plt.title("extracted tumor region")
# else:
#     plt.title("extracted other region")
# plt.xticks([]), plt.yticks([])
# plt.show()

# ---------- Compute GLCM for different directions ----------
# GLCM along x-axis (1 pixel to the right)
glcm_x = graycomatrix(image_rescaled, distances=[1], angles=[0], levels=128, symmetric=True, normed=True)
glcm_x[0, 0, 0, 0] = 0
glcm_x = glcm_x / np.sum(glcm_x)  # normalize

# GLCM along y-axis (1 pixel down)
glcm_y = graycomatrix(image_rescaled, distances=[1], angles=[np.pi/2], levels=128, symmetric=True, normed=True)
glcm_y[0, 0, 0, 0] = 0
glcm_y = glcm_y / np.sum(glcm_y)  # normalize

# ---------- Without function: Compute homogeneity, contrast, uniformity(energy) of GLCM ----------
# Calculate homogeneity, contrast, and uniformity (energy) for glcm_x and glcm_y
homogeneity_x = calculate_homogeneity(glcm_x)
homogeneity_y = calculate_homogeneity(glcm_y)

contrast_x = calculate_contrast(glcm_x)
contrast_y = calculate_contrast(glcm_y)

energy_x = calculate_energy(glcm_x)
energy_y = calculate_energy(glcm_y)

# Print the results
print("---- Without Function ----")
print(f"Homogeneity along x-axis: {homogeneity_x:.4f}")
print(f"Homogeneity along y-axis: {homogeneity_y:.4f}")
print(f"Contrast along x-axis: {contrast_x:.4f}")
print(f"Contrast along y-axis: {contrast_y:.4f}")
print(f"Uniformity (Energy) along x-axis: {energy_x:.4f}")
print(f"Uniformity (Energy) along y-axis: {energy_y:.4f}")

# ---------- Using function: Compute homogeneity, contrast, uniformity(energy) of GLCM ----------
# Homogeneity
homogeneity_x = graycoprops(glcm_x, "homogeneity")[0, 0]
homogeneity_y = graycoprops(glcm_y, "homogeneity")[0, 0]

# Contrast
contrast_x = graycoprops(glcm_x, "contrast")[0, 0]
contrast_y = graycoprops(glcm_y, "contrast")[0, 0]

# Uniformity (Energy)
energy_x = graycoprops(glcm_x, "energy")[0, 0]
energy_y = graycoprops(glcm_y, "energy")[0, 0]

# Print the results
print("---- Using Function ----")
print(f"Homogeneity along x-axis: {homogeneity_x:.4f}")
print(f"Homogeneity along y-axis: {homogeneity_y:.4f}")
print(f"Contrast along x-axis: {contrast_x:.4f}")
print(f"Contrast along y-axis: {contrast_y:.4f}")
print(f"Uniformity (Energy) along x-axis: {energy_x:.4f}")
print(f"Uniformity (Energy) along y-axis: {energy_y:.4f}")

# ---------- Display GLCM ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(glcm_x[:, :, 0, 0], cmap="gray", aspect="auto")
axes[0].set_title("GLCM along x-axis (1 pixel to the right)")
axes[0].set_xlabel("Gray Level")
axes[0].set_ylabel("Gray Level")

axes[1].imshow(glcm_y[:, :, 0, 0], cmap="gray", aspect="auto")
axes[1].set_title("GLCM along y-axis (1 pixel down)")
axes[1].set_xlabel("Gray Level")
axes[1].set_ylabel("Gray Level")

# plt.show()
