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
zscale = visualization.ZScaleInterval()



def get_circular_kernel(radius, halfside):
    x = np.arange(2*halfside + 1) - halfside
    y = np.arange(2*halfside + 1) - halfside
    
    xx,yy=np.meshgrid(x,y)
    
    base = ( yy**2 + xx**2 <= radius**2)
    
    return base.astype('int')




def get_petrosian_radius(row, col, image,mask):
    # if image_patch.size[0]< 36:
        # raise Exception('The patch should be larger than 36 x 36 !')
    shape = image.shape
    row, col = int(row), int(col)
    print(f'row: {row}, col: {col}')
    
    petrosian_radius = 4
    outer_radius_low = np.floor(petrosian_radius * 0.8).astype('int')
    outer_radius_high = np.ceil(petrosian_radius * 1.2).astype('int')
    
    halfside = 36 # maximum outer radius

    
    print(outer_radius_low.dtype)
    
    patch = image[row - halfside : row + halfside + 1,\
                      col - halfside : col + halfside + 1]
    
    patchmask = mask[row - halfside : row + halfside + 1,\
                      col - halfside : col + halfside + 1]
    
    
    x = np.arange(2*halfside + 1) - halfside
    y = np.arange(2*halfside + 1) - halfside
    
    xx,yy=np.meshgrid(x,y)
    
    
    inner_circle = get_circular_kernel(petrosian_radius,halfside)
    
    
    outer_low_circle = get_circular_kernel(outer_radius_low,halfside)
    outer_high_circle = get_circular_kernel(outer_radius_high,halfside)
    outer_crown = np.absolute(outer_low_circle - outer_high_circle)

    innerpatch = (patch * inner_circle)
    innervalues = innerpatch[innerpatch.nonzero()]
    
    outerpatch = (patch * outer_crown)
    outervalues = outerpatch[outerpatch.nonzero()]
    
    
    print('Inner mean: ', np.mean(innervalues))
    
    print('Outer mean: ', np.mean(outervalues))

    plt.figure()
    # plt.subplot(1,2,1)
    # imshow(inner_circl)
    imshow(inner_circle,alpha = 0.4)
    imshow(outer_crown,alpha=0.4,cmap='seismic')
    plt.axvline(halfside)
    plt.axhline(halfside)
    
    plt.figure()
    imshow(zscale(patch))
    
    plt.figure()
    imshow(patchmask)

# img = np.ones((40, 40))
# get_petrosian_radius(300,300,image)

#%%
#Import actual image
filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()

#Import convolution threshold mask
filename = "mask.fits" # with frame but no star
hdulist=fits.open(filename)
mask = hdulist[0].data
hdulist.close()

# import galaxylist
galaxylist = np.loadtxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt')
# galaxylist = np.loadtxt('located_galaxies_00/galaxypositions-final.txt')

gal = galaxylist[700] # 1000?

get_petrosian_radius(gal[0], gal[1], image, mask)




