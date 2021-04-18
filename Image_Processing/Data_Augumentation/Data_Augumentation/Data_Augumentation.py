
##from functions import *



t_image_path = "E:/!!LIBRARIES/Python/fastai/images/0.jpg"
##img = load_img(t_image_path)

##show_img(img, title="Original")


##t_img_rot90 = rotate_img(img, 90.0)

##show_img(t_img_rot90, title="Rotated 90°")



#from fastai.vision import *

#from fastai.vision.augment import rotate, Rotate


#from fastai.vision.augment import rotate

#tfms = [rotate(p=1.,draw=180)]


### Function that displays many transformations of an image
##def plots_of_one_image(img_path, tfms, rows=1, cols=3, width=15, height=5, **kwargs):
##    img = get_img(img_path)
##    [img.apply_tfms(tfms, **kwargs).show(ax=ax) 
##     for i,ax in enumerate(plt.subplots(rows,cols,figsize=(width,height))[1].flatten())]           
    

##with no_random():
##    thetas = [-30,-15,0,15,30]
##    imgs = _batch_ex(5)
##    deflt = Rotate()
##    const = Rotate(p=1.,draw=180) #same as a vertical flip
##    listy = Rotate(p=1.,draw=[-30,-15,0,15,30]) #completely manual!!!
##    funct = Rotate(draw=lambda x: x.new_empty(x.size(0)).uniform_(-10, 10)) #same as default

##    show_images( deflt(img) ,suptitle='Default Rotate, notice the small rotation',titles=[i for i in range(imgs.size(0))])
##    show_images( const(img) ,suptitle='Constant 180 Rotate',titles=[f'180 Degrees' for i in range(imgs.size(0))])
##    #manually specified, not random! 
##    show_images( listy(img) ,suptitle='Manual List Rotate',titles=[f'{i} Degrees' for i in [-30,-15,0,15,30]])
##    #same as default
##    show_images( funct(img) ,suptitle='Default Functional Rotate',titles=[i for i in range(imgs.size(0))])









#ALBUMENTATIONS!!!

#import random
#import cv2
#from matplotlib import pyplot as plt
#import albumentations as A


#def view_transform(image):
#    plt.figure(figsize=(5, 5))
#    plt.axis('off')
#    plt.imshow(image)
#    plt.show()


#figure = cv2.imread(t_image_path)
#figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
#view_transform(figure)


#transform = A.HorizontalFlip(p=0.5)
#random.seed(7)
#augmented_image = transform(image=figure)['image']
#view_transform(augmented_image)





##################################################################
##################################################################
##################################################################
##################################################################
##################################################################


from fastai.data.all import *
from fastai.vision.core import *
from fastai.vision.data import *
from fastai.vision.augment import *

#from nbdev.showdoc import *

import torch as t
import numpy as np
import torchvision.transforms.functional as TF



def load_img(Path:str):
    '''
        Args:
            Path: string which contains a path to specified image

        Return:
            TensorImage [N,C,H,W]
    '''
    image = Image.open(Path)
    x = TensorImage(image).permute(2,0,1).float()/255.
    return x.unsqueeze_(0)


def show_img(img, **kwargs):    
    '''
        Args:
            img: image to show, i.e PILImage etc.

        Return:
            None
    '''
    show_image(img.squeeze(0), **kwargs)
    plt.show() 




t_image_path = "E:/!!LIBRARIES/Python/fastai/images/0.jpg"


#img = load_img(t_image_path)
#show_img(img, title='original')


@patch
def flip_lr(x:TensorImageBase):
    return x.flip(-1)



#t_img = img.flip_lr()

#show_img(img, title="Original shape {0}".format(img.shape))
#show_img(t_img, title="Flip left right {0}".format(t_img.shape))


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################







img = load_img(t_image_path)
print(img.shape)
show_img(img[0], title="Original")




#const_rot_90 = Rotate(p=1.,draw=90) #rotate 90 degrees
#timg_rotate90 = const_rot_90(img.float())
#timg_rotate90 = timg_rotate90.squeeze(dim=0)
#print(timg_rotate90.shape)
#show_img(timg_rotate90, title="Rotated 90 degrees")



#timg_permuted = img.permute(0,1,3,2)
#timg_permuted = timg_permuted.squeeze(dim=0)
#print(timg_permuted.shape)
#show_img(timg_permuted, title="Permuted")


#timg_transposed = t.transpose(img, 2,3)
#timg_transposed = timg_transposed.squeeze(dim=0)
#print(timg_transposed.shape)
#show_img(timg_transposed, title="Transposed {0}".format(timg_transposed.shape))




const_flip_img = Flip(p=1.,draw=1)
timg_flipped = const_flip_img(img.float())
print(timg_flipped.shape)
show_img(timg_flipped, title="Flipped {0}".format(timg_flipped.shape))

#test_eq(const_flip_img(img,split_idx=0), tensor([[1.,0., 0.,1]]) -1)