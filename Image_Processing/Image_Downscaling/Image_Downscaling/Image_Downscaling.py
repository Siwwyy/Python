
import matplotlib.pyplot as plt
import cv2

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



from PIL import Image
import numpy as np
import skimage


img = Image.open(PATH)
img.load()

img_array = np.asarray(img, dtype='int32')

#red1 = np.copy(img_array)
#red1[:,:,0] = 255 # this is on (top row below)
#plt.imshow(red1)
#plt.show()

#blue0 = np.copy(img_array)
#blue0[:,:,2] = 0 # this is off (bottom row below)
#plt.imshow(blue0)

#plt.show()

downsample = 20
# first, change to 0-1
ds_array = img_array / 255

r = skimage.measure.block_reduce(ds_array[:, :, 0],
                                 (downsample, downsample),
                                 np.mean)
g = skimage.measure.block_reduce(ds_array[:, :, 1],
                                 (downsample, downsample),
                                 np.mean)
b = skimage.measure.block_reduce(ds_array[:, :, 2],
                                 (downsample, downsample),
                                 np.mean)
ds_array = np.stack((r, g, b), axis=-1)



fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()

ax[0].imshow(img_array)
ax[0].set_title("Oryginal")
ax[0].set_xlabel(str(img_array.shape))

ax[1].imshow(ds_array)
ax[1].set_title("Downsampled by {0}x{0}".format(downsample))
ax[1].set_xlabel(str(ds_array.shape))


plt.tight_layout()


#plt.imshow(img_array)


figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()