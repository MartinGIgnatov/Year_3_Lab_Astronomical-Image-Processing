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

def show(img):
    plt.figure()
    imshow(img)
    plt.show()
    
    
    
zscale=visualization.ZScaleInterval()

# import image
filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()

filename = "mask.fits" # with frame but no star
hdulist=fits.open(filename)
mask = hdulist[0].data
hdulist.close()

# # show(zscale(image))
# top=1742; bottom = 1997; left = 1862; right = 2104
# patch=image[1742:1997,1862:2104]
# patchscale=mask[1742:1997,1862:2104]
# # imscale(patch)
# show(zscale(patch))
# show(patchscale)
#%%
top=1742; bottom = 1997; left = 1862; right = 2104
patch=image[1742:1997,1862:2104]
patchmask=mask[1742:1997,1862:2104]
extent=np.array([left,right,bottom,top])-0.5

"""
plt.subplot(1,2,1)
imshow(zscale(patch),extent=extent)
plt.subplot(1,2,2)
imshow(patchmask,extent=extent)
plt.show()
"""
# show(patchmask)

# import galaxy list
galaxylist_raw=np.loadtxt('located_galaxies_00/galaxypositions-withcorner.txt')
galaxylist=[]

for galaxy in galaxylist_raw:
    row, col, maxpix, numpix = galaxy
    if row > top and row < bottom:
        if col > left and col < right:
            if maxpix > 3465 and maxpix < 35000:
                galaxylist.append(galaxy)
                
    
radius_inner = 10
radius_outer = 20

for galaxy in galaxylist:
    row, col, maxpix, numpix = galaxy
    
    print(row, col)
    
    if ( row - radius_outer < 0 or  - row - radius_outer < 0 ) or\
       ( col - radius_outer < 0 or image.shape[1] - col - radius_outer < 0 ):
           continue
    
    x = np.arange(image.shape[1]) - col
    y = np.arange(image.shape[0]) - row
    xx,yy=np.meshgrid(x,y)
    
    base_inner = np.zeros(image.shape)
    base_outer = np.zeros(image.shape)
    
    base_inner = ( yy**2 - xx**2 < radius_inner )
    base_outer = ( yy**2 - xx**2 < radius_outer ) - ( yy**2 - xx**2 < radius_inner )
    
    show(base_inner)
    show(base_outer)
    
    input()


            
galaxylist=np.array(galaxylist)
print(f'galaxylist: {len(galaxylist)} galaxies,\n{galaxylist}')

np.savetxt(f'located_galaxies_00/galaxylist_{top}_{bottom}_{left}_{right}.txt', galaxylist,\
            header = 'top bottom left right')
    
tiff.imsave(f'located_galaxies_00/mask_{top}_{bottom}_{left}_{right}.tiff', patchmask)
tiff.imsave(f'located_galaxies_00/patch_{top}_{bottom}_{left}_{right}.tiff',zscale(patch))
