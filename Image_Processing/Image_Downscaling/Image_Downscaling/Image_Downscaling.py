
import matplotlib.pyplot as plt
import cv2
import numpy as np

from PIL import Image
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

PATH = "4K_1.jpg"

#img_array = cv2.imread(PATH,cv2.IMREAD_COLOR)

#print(img_array.shape)
#image_downscaled = downscale_local_mean(img_array, (4,4,1))

#fig, axes = plt.subplots(nrows=2, ncols=1)

#ax = axes.ravel()

#ax[0].imshow(img_array)
#ax[0].set_title("Original image")

#ax[1].imshow(image_downscaled / 255.0)
#ax[1].set_title("Downscaled image (no aliasing)")

#plt.tight_layout()
#plt.show()


#plt.figure(figsize=(1000,1000))
#plt.imshow(img_array[...,::-1])
##plt.tight_layout()
#plt.show()

####################################################################
####################################################################
####################################################################
####################################################################



#from PIL import Image
#import numpy as np
#import skimage


#img = Image.open(PATH)
#img.load()

#img_array = np.asarray(img, dtype='int32')

#red1 = np.copy(img_array)
#red1[:,:,0] = 255 # this is on (top row below)
#plt.imshow(red1)
#plt.show()

#blue0 = np.copy(img_array)
#blue0[:,:,2] = 0 # this is off (bottom row below)
#plt.imshow(blue0)

#plt.show()

#downsample = 8
## first, change to 0-1
#ds_array = img_array / 255

#r = skimage.measure.block_reduce(ds_array[:, :, 0],
#                                 (downsample, downsample),
#                                 np.mean)
#g = skimage.measure.block_reduce(ds_array[:, :, 1],
#                                 (downsample, downsample),
#                                 np.mean)
#b = skimage.measure.block_reduce(ds_array[:, :, 2],
#                                 (downsample, downsample),
#                                 np.mean)
#ds_array = np.stack((r, g, b), axis=-1)



#fig, axes = plt.subplots(nrows=1, ncols=2)
#ax = axes.ravel()

#ax[0].imshow(img_array)
#ax[0].set_title("Oryginal")
#ax[0].set_xlabel(str(img_array.shape))

#ax[1].imshow(ds_array)
#ax[1].set_title("Downsampled by {0}x{0}".format(downsample))
#ax[1].set_xlabel(str(ds_array.shape))


#plt.tight_layout()


##plt.imshow(img_array)


#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
#plt.show()




##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################


def Data_Decimation(image, kernel_size):
    try:
        downsampled_image = np.zeros((image.shape[0] // kernel_size[0],image.shape[1] // kernel_size[1], image.shape[2]), dtype = np.int32)
    except:
        print("Image has to have a three dimensions, x,y and channels e.g: If It is in RGB scale then shape should looks like: (x,y,3)")

    for channels in range(image_data.shape[2]):
        for i in range(downsampled_image.shape[0]):
            for j in range(downsampled_image.shape[1]):
                downsampled_image[i][j][channels] = image[i * kernel_size[0]][j * kernel_size[1]][channels]
                #print(i * kernel_size[0], j * kernel_size[1], sep = " | ")

    return downsampled_image



Kernel_Size = (24,24)

image = Image.open(PATH)
# convert image to numpy array
image_data = np.asarray(image, dtype='uint8')
#image_data = np.zeros((6,6,3), dtype='int32')
Downsampled_Image = Data_Decimation(image_data,Kernel_Size)


fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()

ax[0].imshow(image_data)
ax[0].set_title("Oryginal")
ax[0].set_xlabel(str(image_data.shape))

ax[1].imshow(Downsampled_Image)
ax[1].set_title("Downsampled by {0}x{0}".format(Kernel_Size[0]))
ax[1].set_xlabel(str(Downsampled_Image.shape))


plt.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

Downsampled_Image = Image.fromarray(Downsampled_Image.astype(np.uint8))
Downsampled_Image.save("Downsampled_{0}x{0}.jpg".format(Kernel_Size[0]))