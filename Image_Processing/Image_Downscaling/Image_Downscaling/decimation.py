import numpy as np
import cv2
import torch

def image_decimation(img, kernel_size, anchor, dtype=torch.float):
    """Decimate a .exr file by given scale
    
        Args:
            img: input photo to decimate 
            kernel_size: determines a matrix to decimate (how rescale the image will be)
            anchor: position of pixel around of center of the image
            # 00 01
            # 10 11
    """
    
    size = img.shape
    decimated_tensor = torch.zeros((size[0] // kernel_size[0], size[1] // kernel_size[1], size[2]), dtype=dtype) 

    step_x = kernel_size[0]
    step_y = kernel_size[1]

    offset_x = (kernel_size[0] // 2) - 1
    offset_y = (kernel_size[1] // 2) - 1
    

    #decimated_tensor = torch.arange(-0.25, 64-0.25, 0.5, dtype=dtype)
    shape = img.shape
    h,w, _ = img.shape
    
    for channel in range(img.shape[2]):
        x = torch.arange(start=img[0][0][channel], end=img[0][-1][channel], step=kernel_size[0], dtype=dtype)
        decimated_tensor[:][:][channel] = x

    #y = torch.arange(shape[0], dtype=dtype)
    #meshgrid_y, meshgrid_x = torch.meshgrid(y, x)


    #for channel in range(decimated_tensor.shape[2]): #Channel e.g: R G B, then
    #we have a 3-channel image
    #            decimated_tensor[i][j][channel] = img[i* step_x + offset_x +
    #            anchor[0]][j * step_y + offset_y + anchor[1]][channel]

    #step_x = kernel_size[0]
    #step_y = kernel_size[1]

    #offset_x = (kernel_size[0] // 2) - 1
    #offset_y = (kernel_size[1] // 2) - 1
    
    #for i in range(decimated_tensor.shape[0]):
    #    for j in range(decimated_tensor.shape[1]):
    #        for channel in range(decimated_tensor.shape[2]): #Channel e.g: R G
    #        B, then we have a 3-channel image
    #            decimated_tensor[i][j][channel] = img[i * step_x + offset_x +
    #            anchor[0]][j * step_y + offset_y + anchor[1]][channel]


    return decimated_tensor
    
    #return torch.stack([meshgrid_x, meshgrid_y], axis=-1) # stack 2 channels: (y,x)


#def grid_sample_pixel_coords(img_data, xy_coords, mode='bilinear', padding_mode="border",
#                             align_corners=True, xy_offset=None):
#    """Sample the input image along the grid of xy coordinates (expressed in pixels).
#        Args:
#            img_data: (N,C,H,W)
#            xy_coords: (N,H,W,2), the last axis are the coordinates expressed in pixels
#                ([0,img_width-1], [0,img_height-1])
#            mode: interpolation mode to calculate output values 'bilinear' | 'nearest' | 'bicubic'.
#                See F.grid_sample().
#            padding_mode="border": to handle cases where we slightly exceed the edge pixel's center.
#            align_corners=True: otherwise grid wouldn't line up.
#            xy_offset: a 2-tensor constant offset (e.g. -jitter) expressed in pixels.
#        Return:
#            output tensor: (N,C,H,W)
#    """
#    return F.grid_sample(img_data, pixel_to_img_coords(xy_coords, img_data, xy_offset=xy_offset), mode=mode,
#                         padding_mode=padding_mode, align_corners=align_corners)