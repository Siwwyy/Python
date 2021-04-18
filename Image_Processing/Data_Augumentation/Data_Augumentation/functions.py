import torch as t
import numpy as np

from fastai.data.all import *
from fastai.vision.core import *
from fastai.vision.data import *
#from fastai.vision.augment import rotate
from fastai.vision.all import *
#from fastai.vision
#from nbdev.showdoc import *


def load_img(Path:str):
    '''
        Args:
            Path: string which contains a path to specified image

        Return:
            PILImage image
    '''
    return PILImage(PILImage.create(Path))

def show_img(img, **kwargs):    
    '''
        Args:
            img: image to show, i.e PILImage etc.

        Return:
            None
    '''
    show_image(img, **kwargs)
    plt.show() 


def rotate_img(img, degrees:torch.float16): 
    tfms = Rotate(max_deg=degrees, p=1.0)
    #img_rotated = img.apply_tfms(tfms)
    return tfms(img)