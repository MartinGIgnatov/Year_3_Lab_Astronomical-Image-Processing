import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from astropy.io import fits
import sys
from kernels import convolve,EDGEDETECTION,GAUSSIANBLUR,gaussian
import astropy.visualization as visualization
import time
from sys import stdout
import tifffile as tiff
import galaxy_list_filter as galaxyfilter
plt.style.use('mystyle-2.mplstyle')



def get_circular_kernel(radius):
    x = np.arange(2*radius + 1) - radius
    y = np.arange(2*radius + 1) - radius
    
    xx,yy=np.meshgrid(x,y)
    
    base = ( yy**2 + xx**2 <= radius**2 )
    
    return base.astype('int')




def get_petrosian_radius(col, row, image_patch, background_patch):
    # if image_patch.size[0]< 36:
        # raise Exception('The patch should be larger than 36 x 36 !')
    size = image_patch.size
    
    petrosian_radius = 3
    outer_low = np.floor(petrosian_radius * 0.8)
    outer_high = np.ceil(petrosian_radius * 1.2)
    
    x = np.arange(2*radius + 1) - radius
    y = np.arange(2*radius + 1) - radius
    
    xx,yy=np.meshgrid(x,y)
    
    
    inner_circle = get_circular_kernel(petrosian_radius)
    
    outer_low_circle = get_circular_kernel(outer_low)
    outer_low_circle = get_circular_kernel(outer_high)
    outer_crown = (outer_low_circle)
    
    imshow(circle)

get_petrosian_radius(1,2,3,4)
    