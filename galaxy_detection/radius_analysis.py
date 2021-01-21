import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from astropy.io import fits
from kernels import convolve,EDGEDETECTION,GAUSSIANBLUR,gaussian
import astropy.visualization as visualization
import time
from sys import stdout
import tifffile as tiff
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


top=1742; bottom = 1997; left = 1862; right = 2104
"""
patch = image[1742:1997,1862:2104]
patc_laplacian_treshold = laplacian_treshold[1742:1997,1862:2104]
extent=np.array([left,right,bottom,top])-0.5
"""
"""
plt.subplot(1,2,1)
imshow(zscale(patch),extent=extent)
plt.subplot(1,2,2)
imshow(patchmask,extent=extent)
plt.show()
"""
# show(patchmask)

# import galaxy list
galaxylist_raw = np.loadtxt('located_galaxies_00/galaxypositions-withcorner.txt')
galaxylist = []

radius_inner = 15
radius_outer = 30

for galaxy in galaxylist_raw:
    row, col, maxpix, numpix = galaxy
    if row > top and row < bottom:
        if col > left and col < right:
            if ( row - radius_outer < 0 or image.shape[0] - row - radius_outer < 0 ) or\
                ( col - radius_outer < 0 or image.shape[1] - col - radius_outer < 0 ):
                continue
            if maxpix > 3465 and maxpix < 35000:
                galaxylist.append(galaxy)
                
    

background_image = np.copy(image)


for num, galaxy in enumerate(galaxylist):
    
    print(f" {num} of {len(galaxylist)}")
    
    row, col, maxpix, numpix = galaxy
    
    row = int(row)
    col = int(col)
    
    x = np.arange(image.shape[1]) - col
    y = np.arange(image.shape[0]) - row
    
    xx,yy=np.meshgrid(x,y)
    
    base = ( yy**2 + xx**2 > radius_inner**2 )
    
    background_image = background_image * base

#%%

galaxy_intensities = []
galaxy_background = []


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
    
    
    galaxy_intensities.append(intensity)
    galaxy_background.append(background)
    
    
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
