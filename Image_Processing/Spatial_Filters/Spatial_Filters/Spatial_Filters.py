#import cv2
#import copy
#import random
#import imutils
#import numpy as np

#img = cv2.imread('template.jpg')


#img_blur = cv2.blur(img, (3,3))
#img_box = cv2.boxFilter(img, -1,(5,5), normalize = True)
## src - input image
## ksize - Kernel size
## ddepth - the output image depth (Pass -1 to use that of input)

#cv2.imshow('origin image', imutils.resize(img, 500))
#cv2.imshow('image blur', imutils.resize(img_blur, 500))
#cv2.imshow('image box', imutils.resize(img_box, 500))

#if cv2.waitKey(0) == 27:
#    cv2.destroyAllWindows()
##cv2.resize(img_hr, dsize=(width,height), interpolation=cv2.INTER_AREA) # low-res image


import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


image = cv2.imread('template.jpg') # reads the image
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # convert to HSV
figure_size = 3 # the dimension of the x and y axis of the kernal.
new_image = cv2.blur(image,(figure_size, figure_size))


plt.figure(figsize=(11,6))
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_HSV2RGB)), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)), plt.title('Mean filter')
plt.xticks([]), plt.yticks([])
plt.show()