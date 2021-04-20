
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
    #plt.show() 




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




const_rot_90 = Rotate(p=1.,draw=90) #rotate 90 degrees
timg_rotate90 = const_rot_90(img.float())
timg_rotate90 = timg_rotate90.squeeze(dim=0)
print(timg_rotate90.shape)
show_img(timg_rotate90, title="Rotated 90 degrees")



timg_permuted = img.permute(0,1,3,2)
timg_permuted = timg_permuted.squeeze(dim=0)
print(timg_permuted.shape)
show_img(timg_permuted, title="Permuted")


timg_transposed = t.transpose(img, 2,3)
timg_transposed = timg_transposed.squeeze(dim=0)
print(timg_transposed.shape)
show_img(timg_transposed, title="Transposed {0}".format(timg_transposed.shape))




const_flip_img = Flip(p=1.,draw=1)
timg_flipped = const_flip_img(img.float())
print(timg_flipped.shape)
show_img(timg_flipped, title="Flipped {0}".format(timg_flipped.shape))



y = img.brightness(draw=0.9, p=1.)
show_img(y, title="Brightness {0}".format(y.shape))

a = img.contrast(draw=2., p=1.)
show_img(a, title="Contrast {0}".format(a.shape))


_,axs = subplots(2, 4)
for i,ax in enumerate(axs.flatten()):
    #print(i)
    t = img.dihedral(k=(i+1))
    show_image(t.squeeze(0), ctx=ax, title="Nr: {0}".format(i+1))

plt.show()
#t = img.dihedral(k=)
#show_img(t, title="Dihedral {0}".format(t.shape))



























#import sys, PIL, matplotlib.pyplot as plt, itertools, math, random, collections, torch
#import scipy.stats, scipy.special

#from enum import Enum, IntEnum
#from torch import tensor, Tensor, FloatTensor, LongTensor, ByteTensor, DoubleTensor, HalfTensor, ShortTensor
#from operator import itemgetter, attrgetter
#from numpy import cos, sin, tan, tanh, log, exp
#from dataclasses import field
#from functools import reduce
#from collections import defaultdict, abc, namedtuple, Iterable
#from typing import Tuple, Hashable, Mapping, Dict

#import mimetypes, abc, functools
#from abc import abstractmethod, abstractproperty




#def affine_grid(size):
#    size = ((1,)+size)
#    N, C, H, W = size
#    grid = FloatTensor(N, H, W, 2)
#    linear_points = torch.linspace(-1, 1, W) if W > 1 else tensor([-1])
#    grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])
#    linear_points = torch.linspace(-1, 1, H) if H > 1 else tensor([-1])
#    grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
#    return grid



#def affine_mult(c, m):
#    if m is None: return c
#    size = c.size()
#    c = c.view(-1,2)
#    c = torch.addmm(m[:2,2], c,  m[:2,:2].t()) 
#    return c.view(size)

#def rotate_img(degrees:torch.float16):
#    angle = degrees * math.pi / 180
#    return [[cos(angle), -sin(angle), 0.],
#            [sin(angle),  cos(angle), 0.],
#            [0.        ,  0.        , 1.]]


##img.affine(rotate_img(30)).show()


##m = rotate_img(30)
##print(m)



##img.show()
##plt.show() 




#########################################################
#########################################################
#########################################################
#########################################################




##class ItemBase():
##    "All transformable dataset items use this type"
##    @property
##    @abstractmethod
##    def device(self): pass
##    @property
##    @abstractmethod
##    def data(self): pass


##class ImageBase(ItemBase):
##    "Img based `Dataset` items derive from this. Subclass to handle lighting, pixel, etc"
##    def lighting(self, func, *args, **kwargs)->'ImageBase': return self
##    def pixel(self, func, *args, **kwargs)->'ImageBase': return self
##    def coord(self, func, *args, **kwargs)->'ImageBase': return self
##    def affine(self, func, *args, **kwargs)->'ImageBase': return self

##    def set_sample(self, **kwargs)->'ImageBase':
##        "Set parameters that control how we `grid_sample` the image after transforms are applied"
##        self.sample_kwargs = kwargs
##        return self
    
##    def clone(self)->'ImageBase': 
##        "Clones this item and its `data`"
##        return self.__class__(self.data.clone())





##class Image(ImageBase):
##    "Supports appying transforms to image data"
##    def __init__(self, px)->'Image':
##        "create from raw tensor image data `px`"
##        self._px = px
##        self._logit_px=None
##        self._flow=None
##        self._affine_mat=None
##        self.sample_kwargs = {}

##    @property
##    def shape(self)->Tuple[int,int,int]: 
##        "Returns (ch, h, w) for this image"
##        return self._px.shape
    
##    @property
##    def size(self)->Tuple[int,int]: 
##        "Returns (h, w) for this image"
##        return self.shape[-2:]
    
##    @property
##    def device(self)->torch.device: return self._px.device
    
##    def __repr__(self): return f'{self.__class__.__name__} ({self.shape})'

##    def refresh(self)->None:
##        "Applies any logit or affine transfers that have been "
##        if self._logit_px is not None:
##            self._px = self._logit_px.sigmoid_()
##            self._logit_px = None
##        if self._affine_mat is not None or self._flow is not None:
##            self._px = grid_sample(self._px, self.flow, **self.sample_kwargs)
##            self.sample_kwargs = {}
##            self._flow = None
##        return self

##    @property
##    def px(self)->TensorImage:
##        "Get the tensor pixel buffer"
##        self.refresh()
##        return self._px
##    @px.setter
##    def px(self,v:TensorImage)->None: 
##        "Set the pixel buffer to `v`"
##        self._px=v

##    @property
##    def flow(self):
##        "Access the flow-field grid after applying queued affine transforms"
##        if self._flow is None:
##            self._flow = affine_grid(self.shape)
##        if self._affine_mat is not None:
##            self._flow = affine_mult(self._flow,self._affine_mat)
##            self._affine_mat = None
##        return self._flow
    
##    @flow.setter
##    def flow(self,v): self._flow=v

##    def lighting(self, func, *args, **kwargs)->'Image':
##        "Equivalent to `image = sigmoid(func(logit(image)))`"
##        self.logit_px = func(self.logit_px, *args, **kwargs)
##        return self

##    def pixel(self, func, *args, **kwargs)->'Image':
##        "Equivalent to `image.px = func(image.px)`"
##        self.px = func(self.px, *args, **kwargs)
##        return self

##    def coord(self, func, *args, **kwargs)->'Image':
##        "Equivalent to `image.flow = func(image.flow, image.size)`"        
##        self.flow = func(self.flow, self.shape, *args, **kwargs)
##        return self

##    def affine(self, func, *args, **kwargs)->'Image':
##        "Equivalent to `image.affine_mat = image.affine_mat @ func()`"        
##        m = tensor(func(*args, **kwargs)).to(self.device)
##        self.affine_mat = self.affine_mat @ m
##        return self

##    def resize(self, size)->'Image':
##        "Resize the image to `size`, size can be a single int"
##        assert self._flow is None
##        if isinstance(size, int): size=(self.shape[0], size, size)
##        self.flow = affine_grid(size)
##        return self

##    @property
##    def affine_mat(self):
##        "Get the affine matrix that will be applied by `refresh`"
##        if self._affine_mat is None:
##            self._affine_mat = torch.eye(3).to(self.device)
##        return self._affine_mat
##    @affine_mat.setter
##    def affine_mat(self,v)->None: self._affine_mat=v

##    @property
##    def logit_px(self):
##        "Get logit(image.px)"
##        if self._logit_px is None: self._logit_px = logit_(self.px)
##        return self._logit_px
##    @logit_px.setter
##    def logit_px(self,v)->None: self._logit_px=v
    
##    def show(self, ax:plt.Axes=None, **kwargs)->None: 
##        "Plots the image into `ax`"
##        show_image(self.px, ax=ax, **kwargs)
    
##    @property
##    def data(self)->TensorImage: 
##        "Returns this images pixels as a tensor"
##        return self.px




#def logit(x:Tensor)->Tensor:  return -(1/x-1).log()
#def logit_(x:Tensor)->Tensor: return (x.reciprocal_().sub_(1)).log_().neg_()

#def contrast(x:Tensor, scale:float)->Tensor: return x.mul_(scale)




#test = Brightness(draw=0.9, p=1.)

#train_ds = load_img(t_image_path)

#train_ds.show()
#plt.show()

#train_ds = test(train_ds)


#train_ds.show()
#plt.show()



t_3x3 = torch.arange(9).view(3,-1).unsqueeze(0)
t_3x3



def transpose(x:Tensor) -> Tensor:
    assert(len(x.shape)) >= 2
    return torch.transpose(x, len(x.shape) - 2, len(x.shape) - 1)


t_3x3_transposed = transpose(t_3x3.clone())
print(t_3x3_transposed)


def rotate(x:Tensor, probability:torch.float16, degrees:torch.float16) -> Tensor:
    if len(x.shape) < 4:
        x.unsqueeze_(0)
    return Rotate(p=probability,draw=degrees)(x)


t_3x3_rotate = rotate(t_3x3.clone(), 1., 90.)
print(t_3x3_rotate)