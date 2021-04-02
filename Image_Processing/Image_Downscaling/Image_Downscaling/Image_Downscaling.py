
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


# def data_decimation(img, kernel_size, anchor, dtype=torch.float32):
#     """Decimate a .exr file by given scale
    
#         Args:
#             img: input photo to decimate
#             kernel_size: determines a matrix to decimate (how rescale the image will be)
#             anchor: position of pixel around of center of the image (x,y)
#             # x,y x,y
#             # 0,0 0,1
#             # 1,0 1,1
#     """
    
#     size = img.shape
#     decimated_tensor = torch.zeros((size[0] // kernel_size[1], size[1] // kernel_size[0], size[2]), dtype=dtype) 
   
#     step_x = kernel_size[0]  #determines kernel step size rightwards x axis of image
#     step_y = kernel_size[1]  #determines kernel step size downwards y axis of image

#     offset_x = (kernel_size[0] // 2) - 1  #determines which id in x axis will be taken for given pixel position
#     offset_y = (kernel_size[1] // 2) - 1  #determines which id in y axis will be taken for given pixel position

#     x = torch.arange(start=img[0][0][0], end=img[0][-1][0], step=step_x, dtype=dtype)

#     print(x)
    
#     # for i in range(decimated_tensor.shape[0]): 
#     #     for j in range(decimated_tensor.shape[1]):
#     #         for channel in range(decimated_tensor.shape[2]):  #Channel e.g: R G B, then we have a 3-channel image
#     #             decimated_tensor[i][j][channel] = img[i * step_y + offset_y + anchor[1]][j * step_x + offset_x + anchor[0]][channel]
       
#     return decimated_tensor


# Kernel_Size = (2,2)
# img = torch.zeros((6,6,1), dtype=torch.uint8)

# Downsampled_Image = data_decimation(img, Kernel_Size, (0,0), torch.uint8)



# 
# # image = Image.open(os.path.join(dir_path, PHOTO_DIR))
# image = cv2.imread(str(os.path.join(dir_path, PHOTO_DIR)),  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
# # my_photo = OpenEXR.InputFile(os.path.join(dir_path, PHOTO_DIR))
# # convert image to numpy array
# #image_data = np.asarray(image, dtype='uint8')



# # print(image_data.shape)
# Downsampled_Image = data_decimation(image, Kernel_Size, (0,0), np.uint8)
# print(image.shape)
# print(Downsampled_Image.shape)

# fig, axes = plt.subplots(nrows=1, ncols=2)
# ax = axes.ravel()

# ax[0].imshow(image)
# ax[0].set_title("Oryginal")
# ax[0].set_xlabel(str(image.shape))

# ax[1].imshow(Downsampled_Image)
# ax[1].set_title("Downsampled by {0}x{0}".format(Kernel_Size[0]))
# ax[1].set_xlabel(str(Downsampled_Image.shape))


# plt.tight_layout()
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.show()


#Downsampled_Image = Image.fromarray(Downsampled_Image.astype(np.uint8))
#Downsampled_Image.save(os.path.join(dir_path,'Downsampled_{0}x{0}.jpg'.format(Kernel_Size[0])))



################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################
################################################################################################################################################



def image_decimation(img, kernel_size, anchor, dtype=torch.float32):
    """Decimate a .exr file by given scale
    
        Args:
            img: input photo to decimate
            kernel_size: determines a matrix to decimate (how rescale the image will be)
            anchor: position of pixel around of center of the image (x,y)
            # x,y x,y
            # 0,0 0,1
            # 1,0 1,1
    """
   
    step_x = kernel_size[0]  #determines kernel step size rightwards x axis of image
    step_y = kernel_size[1]  #determines kernel step size downwards y axis of image

    #find possible pixel indexes from discrete to continuous
    x = torch.linspace(-1, 1, width)
    y = torch.linspace(-1, 1, height)

    #create a offsets
    x_offset = x[(kernel_size[0] // 2) - 1::step_x + anchor[0]]     #determines which id in x axis will be taken for given pixel position
    y_offset = y[(kernel_size[1] // 2) - 1::step_y + anchor[1]]     #determines which id in y axis will be taken for given pixel position

    #mesh grid for pixel indexes of x and y coordinates
    meshx, meshy = torch.meshgrid((x_offset, y_offset))

    #coordinates grid
    grid = torch.stack((meshy, meshx), dim=-1).unsqueeze(0)

    #final output
    decimated_tensor = torch.nn.functional.grid_sample(input, grid, align_corners=True)
    print(decimated_tensor.shape, decimated_tensor, sep='\n')


    return decimated_tensor



def print_grid(grid):
    for i in range(grid.shape[1]):
        print("---------------------------")
        for j in range(grid.shape[2]):
            print("X: {0:.1f} Y: {1:.1f}".format(grid[0][i][j][0],grid[0][i][j][1]))


# def print_input(input_img):
#     for i in range(input_img.shape[2]):
#         _str = " "
#         for j in range(input_img.shape[3]):
#             _str += str(input_img[0][0][i][j]) + ' '
        
#         print(_str)



''' y x 
    00  01  02  03          -1.0|-1.0  -1.0|-0.3  -1.0|0.3   1.0|1.0
    10  11  12  13    =>    -0.3|-1.0  -0.3|-0.3  -0.3|0.3  -0.3|1.0
    20  21  22  23    =>     0.3|-1.0   0.3|-0.3   0.3|0.3   0.3|1.0
    30  31  32  33           1.0|-1.0   1.0|-0.3   1.0|0.3   1.0|1.0

'''


# width = 4
# height = 4
# kernel_size = (2,2)

## print("Input")
#input = torch.arange(width*height).view(1,1,height,width).float()#.view(1, 1, 4, 4).float()
#print(input.shape, input, sep='\n')

#print()
#print()
#print("Casted px indexes")
#x = torch.linspace(-1, 1, width)
#y = torch.linspace(-1, 1, height)
#print(x)
#print(y)
#print()
#print()





#print()
#print()
#print("Offsets")
#x_offset = x[::kernel_size[0]]
#y_offset = y[::kernel_size[1]]
#print(x_offset, y_offset, sep='\n')
#print()
#print()

#print()
#print()
#print("Mesh_x and mesh_y")
#meshx, meshy = torch.meshgrid((x_offset, y_offset))
#print(meshx, meshy, sep='\n')
#print()
#print()



#print()
#print()
#print("Grid")
## grid = torch.stack(torch.meshgrid(torch.linspace(-1,1,kernel_size[0]), torch.linspace(-1,1,kernel_size[0]))[::-1],dim=-1).unsqueeze(0)
## grid1 = torch.stack(torch.meshgrid(torch.linspace(-1,1,4), torch.linspace(-1,1,4))[::-1],dim=-1).unsqueeze(0)
#grid = torch.stack((meshy, meshx), dim=-1).unsqueeze(0)
## grid1 = torch.stack(torch.meshgrid(torch.linspace(-1,1,4), torch.linspace(-1,1,4))[::-1],dim=-1).unsqueeze(0)
#print(grid.shape, grid, sep='\n')
#print()
#print()

##print_grid(grid)

#output = torch.nn.functional.grid_sample(input, grid, align_corners=True)
#print(output.shape, output, sep='\n')



# width = 8
# height = 8
# kernel_size = (8,8)


# #create a input matrix or photo
# input = torch.arange(width*height).view(1,1,height,width).float()
# # print(input)
# # print(input.shape)
# print_input(input)


#Load photo

# Decimated = image_decimation(input, kernel_size, (0,0), torch.uint8)


# #find possible pixel indexes from discrete to continuous
# x = torch.linspace(-1, 1, width)
# y = torch.linspace(-1, 1, height)

# print("Tensor x:",x)
# print("Tensor y:",y)

# #create a offsets
# x_offset = x[1::kernel_size[0]]
# y_offset = y[1::kernel_size[1]]

# #mesh grid for pixel indexes of x and y coordinates
# meshx, meshy = torch.meshgrid((x_offset, y_offset))

# #coordinates grid
# grid = torch.stack((meshy, meshx), dim=-1).unsqueeze(0)

# #final output
# output = torch.nn.functional.grid_sample(input, grid, align_corners=True)
# print(output.shape, output, sep='\n')




# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/0090_prepare_dataset.ipynb (unless otherwise specified).

__all__ = ['pixel_to_img_coords_with_shape', 'get_grid', 'data_decimate', 'halton', 'halton_sequence', 'image_decimate',
           'get_frame_index_from_name', 'save_downscaled_exr_decimate', 'downscale_exr_folder_decimate']

# Cell
from IPython.display import display,HTML
from typing import Any, Callable, Optional, List, Dict, Iterator, Set, Tuple
import shutil

# Cell
#from fastai.vision.all import *
import PIL
import torch
import numpy as np
import Imath, OpenEXR

## Cell
#from fastprogress.fastprogress import master_bar, progress_bar

#from .config import data_folder, DatasetSpec
#from .image import load_exr, save_exr, show_img





# Cell
#this functions is similar to pixel_to_img_coords in 0180_TAA (move it to another notebook)
def pixel_to_img_coords_with_shape(xy_coords, img_width:int, img_height:int, xy_offset=None):
    """Convert xy coordinates expressed in pixels (e.g. from motion vectors) into a range of [-1,1].

        Args:
            xy_coords: (N,H,W,2) where the last axis is an absolute (x,y) coordinate expressed in pixels.
            img_width: image width
            img_height: image height

        Return:
            xy_coords: (N,H,W,2) where the last axis should range between [-1,1], except if the coordinates were out-of-image."""
    if xy_offset is None:
        xy_offset = tensor([0.0, 0.0])

    # TODO: think whether this should be detached...? do we need to propagate gradients?
    xy_coords = xy_coords.clone().detach()

    xy_coords[..., 0:1] = (xy_coords[..., 0:1] + xy_offset[0]) / (img_width-1) * 2 - 1.0 # x coordinates
    xy_coords[..., 1:2] = (xy_coords[..., 1:2] + xy_offset[1]) / (img_height-1) * 2 - 1.0 # y coordinates
    return xy_coords

# Cell
def get_grid(shapeHr, KernelSize, anchor):
    """Creates a grid for xy coordinates of pixels into range of [-1,1]

        Args:
            shapeHr: shape of given image (High resolution image)
            KernelSize: size of kernel in (x,y) coordinates manner
            anchor: position of pixel around of center of the kernel size

        Return:
            xy_coords: (N,H,W,2) where the last axis should range between [-1,1], except if the coordinates were out-of-image.
    """
    x = torch.arange(start = 0, end = shapeHr[-1], dtype = torch.float32)
    y = torch.arange(start = 0, end = shapeHr[-2], dtype = torch.float32)
    #create xy offsets
    x_offset = x[KernelSize[0] // 2 - 1 + anchor[0]::KernelSize[0]]
    y_offset = y[KernelSize[1] // 2 - 1 + anchor[1]::KernelSize[1]]
    #mesh grid for pixel indexes of x and y coordinates
    meshy, meshx = torch.meshgrid((y_offset, x_offset))
    #created mesh
    stacked_mesh = torch.stack([meshx, meshy], axis=-1).unsqueeze(0)

    #coordinates grid
    return pixel_to_img_coords_with_shape(stacked_mesh, shapeHr[-1], shapeHr[-2])

# Cell
def data_decimate(tensorHr, KernelSize, anchor):
    """Decimates a given image

        Args:
            tensorHr: high resolution torch tensor (HR image)
            KernelSize: size of kernel in (x,y) coordinates manner
            anchor: position of pixel around of center of the kernel size

        Return:
            TensorImage: decimated tensor
    """
    n = tensorHr.shape[0]
    w = tensorHr.shape[2] // KernelSize[0]
    h = tensorHr.shape[1] // KernelSize[1]
    grid = get_grid(tensorHr.shape, KernelSize, anchor)
    tensor = F.grid_sample(tensorHr.unsqueeze(0).float(), grid, mode='nearest')

    return TensorImage(tensor.squeeze(0))


# Cell
def halton(index, base):
    """Creates a halton index value

        Args:
            index: number/index of sequence
            base: base of given sequence

        Return:
            result: value of given sequence
    """
    f = 1.0
    result = 0.0
    i = index
    while i > 0:
        f /= base
        result += f * (i % base)
        i = i // base

    return result

# Cell
def halton_sequence(frame_index, KernelSize):
    """Creates a halton sequence

        Args:
            frame_index: index of frame in given video
            KernelSize: size of kernel in (x,y) coordinates manner

        Return:
            (x,y): returns a value of pixel coordinate
    """
    #jitter_index = 1 + (frame_index & 0xf)
    jitter_index = frame_index

    jitter_x = (2 * halton(jitter_index, 2)) - 1
    jitter_y = (2 * halton(jitter_index, 3)) - 1

    #x = int(KernelSize[0] // 2 * jitter_x + 0.5)
    #y = int(KernelSize[1] // 2 * jitter_y + 0.5)

    x = KernelSize[0] // 2 * jitter_x + 0.5
    y = KernelSize[1] // 2 * jitter_y + 0.5

    return (x,y)

# Cell
def image_decimate(tensorHr, KernelSize, frame_index):
    """Decimates an image at given video frame

        Args:
            tensorHr: high resolution torch tensor (HR image)
            KernelSize: size of kernel in (x,y) coordinates manner
            frame_index: frame index in given video

        Return:
             data_decimate: decimated photo from HR to LR
    """
    anchor = halton_sequence(frame_index, KernelSize)

    return data_decimate(tensorHr, KernelSize, anchor)

## Cell
#def get_frame_index_from_name(in_file):
#    res = [int(s) for s in re.findall('\\d+', in_file)]
#    return int(res[0])

## Cell
##this functions is similar to save_downscaled_exr in 0030_image (move it to another notebook)
#import cv2
#def save_downscaled_exr_decimate(in_path:Path, out_path:Path, width:int, height:int, frame_index:int, show=False):
#    """Save a downscaled copy of the EXR image from `in_path`. Save the `width`x`height` result in `out_path`.

#        Assumptions:
#            - Input, output are 16-bit EXR
#            - Width/height aspect ratio is preserved    """

#    img_hr = load_exr(str(in_path))
#    KernelSize = (img_hr.shape[2]//width, img_hr.shape[1]//height)
#    img_lr = image_decimate(img_hr, KernelSize, frame_index) # low-res image
#    save_exr(img_lr, str(out_path))

#    if show: show_img(img_lr, figsize=(15,8))

## Cell
##this functions is similar to downscale_exr_folder in 0030_image (move it to another notebook)
#def downscale_exr_folder_decimate(in_folder:Path, out_folder:Path, width:int, height:int, show=False, jitter=False):
#    """Save a downscaled copy of every .exr image in `in_folder`, into `out_path`, with same filenames.

#        Assumptions: same as `save_downscaled_exr_decimate`"""
#    files = list(in_folder.glob("*.exr"))
#    for i in progress_bar(range(len(files))):
#        in_file = files[i]
#        if jitter:
#            frame_idx = get_frame_index_from_name(str(in_file.name))
#            save_downscaled_exr_decimate(in_file, out_folder/in_file.name, width, height, frame_index, show=show)
#        else:
#            save_downscaled_exr_decimate(in_file, out_folder/in_file.name, width, height, 0, show=show)


############################################################################################
############################################################################################
############################################################################################
############################################################################################





# MY
def test_generate_sequence(kernel_size, length:int):
    """
        Functions generates a list of halton's sequence
    """
    _seq = np.zeros((length, 2), dtype=np.float32)

    for i in range(length):
        _seq[i] = halton_sequence(i, kernel_size)

    return _seq


#TEST

kernel_size = (8,8)
length = 50




sequence = test_generate_sequence(kernel_size, length)


from math import log, floor, ceil, fmod
import numpy as np

def halton(dim, nbpts):
    h = np.empty(nbpts * dim)
    h.fill(np.nan)
    p = np.empty(nbpts)
    p.fill(np.nan)
    P = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    lognbpts = log(nbpts + 1)
    for i in range(dim):
        b = P[i]
        n = int(ceil(lognbpts / log(b)))
        for t in range(n):
            p[t] = pow(b, -(t + 1) )

        for j in range(nbpts):
            d = j + 1
            sum_ = fmod(d, b) * p[0]
            for t in range(1, n):
                d = floor(d / b)
                sum_ += fmod(d, b) * p[t]

            h[j*dim + i] = sum_

    return h.reshape(nbpts, dim)


x = halton(2, length)


fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()

ax[0].plot(sequence[:length, 0], sequence[:length, 1],'o', color='red')
ax[0].set_title("uniform-distribution 1 ({0})".format(length))


#ax[1].figure()
ax[1].plot(x[:length, 0], x[:length, 1],'o', color='red')
ax[1].set_title("uniform-distribution 2 ({0})".format(length))


plt.tight_layout()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.show()

