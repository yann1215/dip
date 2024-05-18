import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn.metrics as skm


# ---------- Initialize ----------
# load nii file
T1w = nib.load("T1w.nii")
T2w = nib.load("T2w.nii")

# choose slice
slice_index = 130

# ---------- Get Data ----------
# get image data
T1w_data = T1w.get_fdata()
T2w_data = T2w.get_fdata()

# print shape
print("Shape of T1w: ", T1w_data.shape)
print("Shape of T2w: ", T2w_data.shape)

T1w_slice = T1w_data[:, slice_index, :]
T2w_slice = T2w_data[:, slice_index, :]
T1w_flat = T1w_slice.flatten()
T2w_flat = T2w_slice.flatten()

# ---------- Plot Image ----------
# T1w slice
plt.imshow(T1w_slice, cmap='gray')
plt.title(f"T1w Slice {slice_index}")
plt.axis("off")
plt.tight_layout()
plt.show()

# T2w slice
plt.imshow(T2w_slice, cmap='gray')
plt.title(f"T2w Slice {slice_index}")
plt.axis("off")
plt.tight_layout()
plt.show()

# ---------- Plot Joint Histogram ----------
# T1w slice 1 & T1w slice 2
plt.hist2d(T1w_flat, T1w_flat, bins=100, norm=colors.LogNorm(), cmap="gray")
plt.colorbar(label="Counts")
plt.xlabel(f"T1w Slice {slice_index}")
plt.ylabel(f"T1w Slice {slice_index}")
plt.title(f"Joint Histogram of T1w Slice {slice_index} and T1w Slice {slice_index}")
plt.tight_layout()
plt.show()

# T1w slice 1 & T2w slice
plt.hist2d(T1w_flat, T2w_flat, bins=100, norm=colors.LogNorm(), cmap="gray")
plt.colorbar(label="Counts")
plt.xlabel(f"T1w Slice {slice_index}")
plt.ylabel(f"T2w Slice {slice_index}")
plt.title(f"Joint Histogram of T1w Slice {slice_index} and T2w Slice {slice_index}")
plt.tight_layout()
plt.show()

# ---------- Calculate Mutual Information ----------
T1w_T2w = skm.mutual_info_score(T1w_flat, T2w_flat)
print("Mutual Information: ", T1w_T2w)
