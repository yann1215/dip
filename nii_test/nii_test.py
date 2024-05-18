import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# 加载NIfTI文件
nii_file = "s001_mask.nii"
img = nib.load(nii_file)

# 获取图像数据
data = img.get_fdata()

# 打印数据的形状
print("数据形状：", data.shape)

# 选择要显示的切片索引
slice_index = 50

# 显示切片
plt.imshow(data[:, :, slice_index], cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.title('Slice {}'.format(slice_index))
plt.show()
