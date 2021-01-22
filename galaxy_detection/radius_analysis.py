import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from astropy.io import fits
from kernels import convolve,EDGEDETECTION,GAUSSIANBLUR,gaussian
import astropy.visualization as visualization
import time
from sys import stdout
import tifffile as tiff
import galaxy_list_filter as galaxyfilter
plt.style.use('mystyle-2.mplstyle')

    
zscale=visualization.ZScaleInterval()

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

radius_inner = 15
radius_outer = 30

ignore_border=150 # avoid galaxies too close to border

# filter out galaxy list
galaxylist_raw = np.loadtxt('located_galaxies_00/galaxypositions-final.txt')
galaxylist = galaxyfilter.clean_list_galaxies(galaxylist_raw,min_brightness=3465,
                                              max_brightness=35000,ignore_border=ignore_border,radius=radius_inner)
    

background_image = np.copy(image)


for num, galaxy in enumerate(galaxylist):
    
    print(f" {num} of {len(galaxylist)}")
    
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


hdu = fits.PrimaryHDU(background_image)
hdu.writeto('background_image.fits')

#%%


filename = "background_image.fits" # with frame but no star
hdulist=fits.open(filename)
background_image = hdulist[0].data
hdulist.close()

show(zscale(background_image))


#%%


galaxy_counts = []


for galaxy in galaxylist:
    row, col, maxpix, numpix = galaxy
    
    row = int(row)
    col = int(col)
    
    x = np.arange(2*radius_outer + 1) - radius_outer
    y = np.arange(2*radius_outer + 1) - radius_outer
    
    xx,yy=np.meshgrid(x,y)
    
    base_inner = ( yy**2 + xx**2 <= radius_inner**2 )
    base_outer = ( yy**2 + xx**2 <= radius_outer**2 ) * ( yy**2 + xx**2 > radius_inner**2 )
    
    
    crop_image = image[row - radius_outer : row + radius_outer + 1,\
                      col - radius_outer : col + radius_outer + 1]
        
    crop_back = background_image[row - radius_outer : row + radius_outer + 1,\
                      col - radius_outer : col + radius_outer + 1]
    
    background = crop_back * base_outer
    background = background.flatten()
    background = np.mean(background[background.nonzero()])
    
    intensity =  crop_image * base_inner - background*(base_inner)
  
    
    intensity = intensity.flatten()
    intensity = np.sum(intensity)
    
    
    galaxy_counts.append(intensity)
    
np.savetxt("Galaxy_Counts.txt", galaxy_counts)
    

#%%


filename = "A1_mosaic.fits" # with frame but no star
hdulist=fits.open(filename)
header = hdulist[0].header
hdulist.close()

galaxy_counts = np.loadtxt("Galaxy_Counts.txt")
galaxy_magnitudes = []

for gal_count in galaxy_counts:
    galaxy_magnitudes.append(header["MAGZPT"] - 2.5 * np.log10(gal_count))

plt.hist(galaxy_magnitudes)

#print(header["MAGZPT"])


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
