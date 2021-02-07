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

# import image
filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()

filename = "mask.fits" # with frame but no star
hdulist=fits.open(filename)
laplacian_treshold = hdulist[0].data
hdulist.close()


def show(img):
    plt.figure()
    imshow(img)
    plt.colorbar()
    plt.show()
    
    
    
def show_gal(image, galaxy, radius_outer):
    row, col, maxpix, numpix = galaxy
    
    row = int(row)
    col = int(col)
    
    x = np.arange(2*radius_outer + 1) - radius_outer
    y = np.arange(2*radius_outer + 1) - radius_outer
    
    xx,yy=np.meshgrid(x,y)
    base = ( yy**2 + xx**2 <= radius_outer**2 )
    
    cut = image[row - radius_outer : row + radius_outer + 1,
                      col - radius_outer : col + radius_outer + 1]
    
    show(zscale(cut))


#%%
# import galaxy list

radius_inner = 14 # 15
radius_outer = 24 # 30

ignore_border=150 # avoid galaxies too close to border

# filter out galaxy list
galaxylist_raw = np.loadtxt('located_galaxies_00/galaxypositions-final.txt')
print(galaxylist_raw[:,0].max())

galaxylist = galaxyfilter.clean_list_galaxies(galaxylist_raw,min_brightness=3465,
                                              max_brightness=35000,ignore_border=ignore_border,radius=radius_inner)

# np.savetxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt',galaxylist,header='row\t col\t maxpix\t no. pix')


# generate black pixel with galaxy in galaxylist marked as white
#%%


background_image = np.copy(image)

background_galaxylist = galaxyfilter.clean_list_galaxies(galaxylist_raw,min_brightness=3465,
                                              max_brightness=70000,ignore_border=ignore_border,radius=radius_inner)

for num, galaxy in enumerate(background_galaxylist):
    
    print(f" {num} of {len(background_galaxylist)}")
    
    row, col, maxpix, numpix = galaxy
    
    row = int(row)
    col = int(col)
    
    rop_image = background_image[row - radius_inner : row + radius_inner + 1,\
                      col - radius_inner : col + radius_inner + 1]
    
    x = np.arange(2*radius_inner + 1) - radius_inner
    y = np.arange(2*radius_inner + 1) - radius_inner
    
    xx,yy=np.meshgrid(x,y)
    
    base_inner = ( yy**2 + xx**2 > radius_inner**2 )
    
    rop_image = rop_image * base_inner
    
    background_image[row - radius_inner : row + radius_inner + 1,\
                      col - radius_inner : col + radius_inner + 1] = rop_image


# hdu = fits.PrimaryHDU(background_image)
# hdu.writeto('galaxy_brightness_analysis_results/background_image.fits')

#%%


filename = "galaxy_brightness_analysis_results/background_image.fits" # with frame but no star
hdulist=fits.open(filename)
background_image = hdulist[0].data
hdulist.close()

show(zscale(background_image))


#%%


galaxy_intensities_mean = []
galaxy_intensities_std = []
galaxy_background_mean = []
galaxy_background_std = []
galaxy_number_inner_pixels = []

for galaxy in galaxylist:
    row, col, maxpix, numpix = galaxy
    
    row = int(row)
    col = int(col)
    
    x = np.arange(2*radius_outer + 1) - radius_outer
    y = np.arange(2*radius_outer + 1) - radius_outer
    
    xx,yy=np.meshgrid(x,y)
    
    base_inner = ( yy**2 + xx**2 <= radius_inner**2 )
    base_outer = ( yy**2 + xx**2 <= radius_outer**2 ) * ( yy**2 + xx**2 > radius_inner**2 )
    
    # take a patch from the immage surrounding the galaxy
    crop_image = image[row - radius_outer : row + radius_outer + 1,\
                      col - radius_outer : col + radius_outer + 1]
    
    # take the corresponding patch frmo the background
    crop_back = background_image[row - radius_outer : row + radius_outer + 1,\
                      col - radius_outer : col + radius_outer + 1]
    
    
    background = crop_back * base_outer
    
    crop_image =  crop_image * base_inner
    
    # intensity = intensity.flatten()
    # background = background.flatten()
    
    
    background_mean = np.mean(background[background.nonzero()])
    background_std = np.std(background[background.nonzero()])
    intensity_mean = np.mean(crop_image[crop_image.nonzero()])
    intensity_std = np.std(crop_image[crop_image.nonzero()])
    buf = crop_image.flatten()
    buf = buf[buf>0]
    number_inner_pixels = len(buf)
    
    galaxy_intensities_mean.append(intensity_mean)
    galaxy_intensities_std.append(intensity_std)
    galaxy_background_mean.append(background_mean)
    galaxy_background_std.append(background_std)
    galaxy_number_inner_pixels.append(number_inner_pixels)
    
#%%
# this is out of the for loop
# put acquired data into columns
galaxydata = np.c_[galaxy_intensities_mean,galaxy_intensities_std,
                   galaxy_background_mean,galaxy_background_std,
                   galaxy_number_inner_pixels]

# np.savetxt('galaxy_brightness_analysis_results/brightness_data.txt',galaxydata,header='intensity_mean \t\
# intensity_std \t backgroudn_mean \t backgroudn_std \t number_inner_pixels')
    

#%%


filename = "../Images/A1_mosaic.fits" # with frame but no star
hdulist=fits.open(filename)
header = hdulist[0].header
hdulist.close()

data = np.loadtxt('galaxy_brightness_analysis_results/brightness_data.txt')

galaxy_counts = (data[:,0] - data[:,2]) * data [:,4]
galaxy_magnitudes = []


print(np.argwhere(galaxy_counts<0))

"""
for gal_count in galaxy_counts:
    if gal_count > 0:
        galaxy_magnitudes.append(header["MAGZPT"] - 2.5 * np.log10(gal_count))

galaxy_magnitudes=np.array(galaxy_magnitudes)
"""

#%%

index = 1

show_gal(image, galaxylist[index], radius_outer)
show_gal(background_image, galaxylist[index], radius_outer)
show_gal(laplacian_treshold, galaxylist[index], radius_outer)

            
buf=np.array(galaxylist)
print(f'galaxylist: {len(buf)} galaxies')

buf=np.array(galaxy_intensities)
print(f'intensities: {len(buf)} int, {buf}')

buf=np.array(galaxy_background)
print(f'background: {len(buf)} back, {buf}')

"""
np.savetxt(f'located_galaxies_00/galaxylist_{top}_{bottom}_{left}_{right}.txt', galaxylist,\
            header = 'top bottom left right')
    
tiff.imsave(f'located_galaxies_00/mask_{top}_{bottom}_{left}_{right}.tiff', patchmask)
tiff.imsave(f'located_galaxies_00/patch_{top}_{bottom}_{left}_{right}.tiff',zscale(patch))
"""
