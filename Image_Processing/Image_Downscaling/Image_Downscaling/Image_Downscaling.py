
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


def data_decimation(img, kernel_size, anchor, dtype=torch.float32):
    """Decimate a .exr file by given scale
    
        Args:
            img: input photo to decimate
            kernel_size: determines a matrix to decimate (how rescale the image will be)
            anchor: position of pixel around of center of the image (x,y)
            # x,y x,y
            # 0,0 0,1
            # 1,0 1,1
    """
    
    size = img.shape
    decimated_tensor = torch.zeros((size[0] // kernel_size[1], size[1] // kernel_size[0], size[2]), dtype=dtype) 
   
    step_x = kernel_size[0]  #determines kernel step size rightwards x axis of image
    step_y = kernel_size[1]  #determines kernel step size downwards y axis of image

    offset_x = (kernel_size[0] // 2) - 1  #determines which id in x axis will be taken for given pixel position
    offset_y = (kernel_size[1] // 2) - 1  #determines which id in y axis will be taken for given pixel position

    x = torch.arange(start=img[0][0][0], end=img[0][-1][0], step=step_x, dtype=dtype)

    print(x)
    
    # for i in range(decimated_tensor.shape[0]): 
    #     for j in range(decimated_tensor.shape[1]):
    #         for channel in range(decimated_tensor.shape[2]):  #Channel e.g: R G B, then we have a 3-channel image
    #             decimated_tensor[i][j][channel] = img[i * step_y + offset_y + anchor[1]][j * step_x + offset_x + anchor[0]][channel]
       
    return decimated_tensor


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



def print_grid(grid):
    for i in range(grid.shape[1]):
        print("---------------------------")
        for j in range(grid.shape[2]):
            print("X: {0:.1f} Y: {1:.1f}".format(grid[0][i][j][0],grid[0][i][j][1]))



''' y x 
    00  01  02  03          -1.0|-1.0  -1.0|-0.3  -1.0|0.3   1.0|1.0
    10  11  12  13    =>    -0.3|-1.0  -0.3|-0.3  -0.3|0.3  -0.3|1.0
    20  21  22  23    =>     0.3|-1.0   0.3|-0.3   0.3|0.3   0.3|1.0
    30  31  32  33           1.0|-1.0   1.0|-0.3   1.0|0.3   1.0|1.0

'''


width = 4
height = 4
kernel_size = (2,2)

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



width = 8
height = 8
kernel_size = (4,4)


#create a input matrix or photo
input = torch.arange(width*height).view(1,1,height,width).float()
print(input)

#find possible pixel indexes from discrete to continuous
x = torch.linspace(-1, 1, width)
y = torch.linspace(-1, 1, height)

print(x)
print(y)

#create a offsets
x_offset = x[::kernel_size[0]]
y_offset = y[::kernel_size[1]]

#mesh grid for pixel indexes of x and y coordinates
meshx, meshy = torch.meshgrid((x_offset, y_offset))

#coordinates grid
grid = torch.stack((meshy, meshx), dim=-1).unsqueeze(0)

#final output
output = torch.nn.functional.grid_sample(input, grid, align_corners=True)
print(output.shape, output, sep='\n')