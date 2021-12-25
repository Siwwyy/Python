
from PIL import Image
import numpy as np
import skimage


class Image_Downsample(object):
    """
        Class for Image Downsampling by given times
    """
    __Image_Path = ""
    __Image = 0
    __Downsampled_Image = 0

    def __init__(self, Image):
        super()
        self.__Image = Image

    def Downsample(self, Block_Size, Functor):
        self.__Downsampled_Image = skimage.measure.block_reduce(self.__Image,
                                   Block_Size,
                                   Functor)
        return self.__Downsampled_Image


