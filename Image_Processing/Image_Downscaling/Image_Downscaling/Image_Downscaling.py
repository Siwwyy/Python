
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch
import OpenEXR

from PIL import Image




dir_path = os.path.dirname(os.path.realpath(__file__))
# PHOTO_DIR = "4K_1.jpg"
PHOTO_DIR = "4K_3840x2160.jpg"
# PHOTO_DIR = "00000_3840x2160.exr"


def data_decimation(img, size, kernel_size, anchor, dtype=np.float32):
    """Decimate a .exr file by given scale
    
        Args:
            img: input photo to decimate 
            size: shape of img
            kernel_size: determines a matrix to decimate (how rescale the image will be)
            anchor: position of pixel around of center of the image
            # x,y x,y
            # 0,0 0,1
            # 1,0 1,1
    """
    
    decimated_tensor = np.zeros((size[0] // kernel_size[0], size[1] // kernel_size[1], size[2]), dtype=dtype) 
   
    step_x = kernel_size[0]  #determines kernel step size rightwards x axis of image
    step_y = kernel_size[1]  #determines kernel step size downwards y axis of image

    offset_x = (kernel_size[0] // 2) - 1  #determines which id in x axis will be taken for given pixel position
    offset_y = (kernel_size[1] // 2) - 1  #determines which id in y axis will be taken for given pixel position
    
    for i in range(decimated_tensor.shape[0]): 
        for j in range(decimated_tensor.shape[1]):
            for channel in range(decimated_tensor.shape[2]):  #Channel e.g: R G B, then we have a 3-channel image
                decimated_tensor[i][j][channel] = img[i * step_x + offset_x + anchor[0]][j * step_y + offset_y + anchor[1]][channel]

    return decimated_tensor


Kernel_Size = (8,8)
# image = Image.open(os.path.join(dir_path, PHOTO_DIR))
image = cv2.imread(str(os.path.join(dir_path, PHOTO_DIR)),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# my_photo = OpenEXR.InputFile(os.path.join(dir_path, PHOTO_DIR))
# convert image to numpy array
#image_data = np.asarray(image, dtype='uint8')

print(image.shape)

# print(image_data.shape)
Downsampled_Image = data_decimation(image, image.shape, Kernel_Size, (0,0))


fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()

ax[0].imshow(image)
ax[0].set_title("Oryginal")
ax[0].set_xlabel(str(image.shape))

ax[1].imshow(Downsampled_Image)
ax[1].set_title("Downsampled by {0}x{0}".format(Kernel_Size[0]))
ax[1].set_xlabel(str(Downsampled_Image.shape))


plt.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()


#Downsampled_Image = Image.fromarray(Downsampled_Image.astype(np.uint8))
#Downsampled_Image.save(os.path.join(dir_path,'Downsampled_{0}x{0}.jpg'.format(Kernel_Size[0])))