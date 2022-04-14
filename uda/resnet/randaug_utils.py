# -*- coding: utf-8 -*-
# File: randaug_utils.py


import functools
import cv2
import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

from tensorpack import *
from tensorpack.dataflow import imgaug, dataset, AugmentImageComponent
from tensorpack.utils import logger

from imagenet_utils import GoogleNetResize


def randaug_augmentor(isTrain, resize, level=17, max_level=10):
    """
    Augmentor used in fb.resnet.torch, for BGR images in range [0,255].
    """
    if isTrain:
        augmentors = [GoogleNetResize()] if resize else []
        augmentors += [
            #GoogleNetResize(),
            # It's OK to remove the following augs if your CPU is not fast enough.
            # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
            # Removing lighting leads to a tiny drop in accuracy.
            imgaug.RandomOrderAug(
                [# We removed the following augmentation
                 #imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 #imgaug.Contrast((0.6, 1.4), clip=False),
                 #imgaug.Saturation(0.4, rgb=False),
                 #rgb-bgr conversion for the constants copied from fb.resnet.torch
                 #imgaug.Lighting(0.1,
                 #                eigval=np.asarray(
                 #                    [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                 #                eigvec=np.array(
                 #                    [[-0.5675, 0.7192, 0.4009],
                 #                     [-0.5808, -0.0045, -0.8140],
                 #                     [-0.5836, -0.6948, 0.4203]],
                 #                    dtype='float32')[::-1, ::-1]
                 #                )
                #randaug_pool[1]
                ]
            ),
            RandomSelectAug(level, max_level),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


#### The individual augmentors used in UDA ####
class RandomSelectAug(imgaug.ImageAugmentor):
    def __init__(self, level, max_level, num_layer=2):
        self._init(locals())
        fn_pool = [
            autocontrast, 
            brightness,
            color,
            contrast,
            cutout,
            equalize,
            invert,
            posterize,
            rotate,
            sharpness,
            shear_x, 
            shear_y,
            solarize,
            translate_x,
            translate_y,
        ]
        fn_pool += [identity] * len(fn_pool) # RandAug is applied at 50% probability
        self.augment_pool = [
            functools.partial(x, level=level, max_level=max_level) for x in fn_pool
        ]
        self.num_augment = len(fn_pool)

    def _augment(self, img, _):
        # Convert from numpy.array to PIL image
        img = self.numpy2pillow(img)
        # Select augmentor
        idx = np.random.choice(self.num_augment, self.num_layer)
        for ix in idx:
            #prob = np.random.uniform(0.2, 0.8)
            #if np.random.uniform(0., 1.) < prob:
            img = self.augment_pool[ix](img)
        # Convert from PIL image to numpy.array
        out = self.pillow2numpy(img)
        return out

    @staticmethod
    def numpy2pillow(img):
        if img.dtype != 'uint8':
            img = np.uint8(img)
        img = Image.fromarray(img, 'RGB')
        return img
    
    @staticmethod
    def pillow2numpy(img):
        return np.array(img)


def enhance_level_to_arg(level, max_level):
    return (level / max_level) * 1.8 + 0.1


def mult_to_arg(level, max_level, multiplier=1.):
    return int(level / max_level * multiplier)


def rotate_level_to_arg(level, max_level):
    return level / max_level * 30.


def shear_level_to_arg(level, max_level):
    return level / max_level * 0.3


def translate_level_to_arg(level, max_level, multiplier=100.):
    return level / max_level * multiplier


def autocontrast(img, level, max_level):
    del level, max_level
    out = PIL.ImageOps.autocontrast(img)
    return out


def brightness(img, level, max_level):
    arg = enhance_level_to_arg(level, max_level)
    out = PIL.ImageEnhance.Brightness(img).enhance(arg)
    return out


def color(img, level, max_level):
    arg = enhance_level_to_arg(level, max_level)
    out = PIL.ImageEnhance.Color(img).enhance(arg)
    return out


def contrast(img, level, max_level):
    arg = enhance_level_to_arg(level, max_level)
    out = PIL.ImageEnhance.Contrast(img).enhance(arg)
    return out


def cutout(img, level, max_level):
    """
    Applies a mask of size (2 * pad_size, 2 * pad_size) at a random location within img.
    """
    cutout_contrast = 40.
    if not level:
        return img
    pad = mult_to_arg(level, max_level, cutout_contrast)
    w, h = img.size
    # Mask center coordinates
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    # Mask boundary coordinates
    x0 = int(max(0, x0 - pad))
    x1 = int(min(w, x0 + 2 * pad))
    y0 = int(max(0, y0 - pad))
    y1 = int(min(h, y0 + 2 * pad))
    xy = (x0, y0, x1, y1)
    out = img.copy()
    PIL.ImageDraw.Draw(out).rectangle(xy, (128, 128, 128))
    return out
    

def equalize(img, level, max_level):
    del level, max_level
    out = PIL.ImageOps.equalize(img)
    return out


def identity(img, level, max_level):
    del level, max_level
    return img


def invert(img, level, max_level):
    del level, max_level
    out = PIL.ImageOps.invert(img)
    return out


def posterize(img, level, max_level):
    arg = mult_to_arg(level, max_level, 4)
    arg = 8 - arg
    out = PIL.ImageOps.posterize(img, arg)
    return out


def rotate(img, level, max_level):
    arg = rotate_level_to_arg(level, max_level)
    coin = np.round(np.random.uniform(low=0, high=1))
    arg *= coin * 2. - 1.
    out = img.rotate(arg, resample=Image.BICUBIC, fillcolor=(128, 128, 128))
    return out


def sharpness(img, level, max_level):
    arg = enhance_level_to_arg(level, max_level)
    out = PIL.ImageEnhance.Sharpness(img).enhance(arg)
    return out


def shear_x(img, level, max_level):
    arg = shear_level_to_arg(level, max_level)
    coin = np.round(np.random.uniform(low=0, high=1))
    arg *= coin * 2. - 1.
    out = img.transform(img.size, PIL.Image.AFFINE, (1, arg, 0, 0, 1, 0), 
        Image.BICUBIC, fillcolor=(128, 128, 128)
    )
    return out


def shear_y(img, level, max_level):
    arg = shear_level_to_arg(level, max_level)
    coin = np.round(np.random.uniform(low=0, high=1))
    arg *= coin * 2. - 1.
    out = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, arg, 1, 0), 
        Image.BICUBIC, fillcolor=(128, 128, 128)
    )
    return out


def solarize(img, level, max_level):
    #arg = mult_to_arg(level, max_level, 256)
    del level, max_level
    arg = 128
    out = PIL.ImageOps.solarize(img, arg)
    return out


def translate_x(img, level, max_level):
    translate_const = 100.
    arg = translate_level_to_arg(level, max_level, translate_const)
    coin = np.round(np.random.uniform(low=0, high=1))
    arg *= coin * 2. - 1.
    arg = int(arg * img.size[0])
    out = img.transform(img.size, PIL.Image.AFFINE, (1, 0, arg, 0, 1, 0), 
        Image.BICUBIC, fillcolor=(128, 128, 128)
    )
    return out


def translate_y(img, level, max_level):
    translate_const = 100.
    arg = translate_level_to_arg(level, max_level, translate_const)
    coin = np.round(np.random.uniform(low=0, high=1))
    arg *= coin * 2. - 1.
    arg = int(arg * img.size[1])
    out = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, arg),
        Image.BICUBIC, fillcolor=(128, 128, 128)
    )
    return out
