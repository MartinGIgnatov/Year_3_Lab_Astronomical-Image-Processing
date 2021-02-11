import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from astropy.io import fits
from PIL import Image
import astropy.visualization as visualization
plt.style.use('mystyle-2.mplstyle')
zscale=visualization.ZScaleInterval()


# load no-star image
maskpath='../Images/star_mask.fits' # image with central star and blooming removed
hdulist=fits.open(maskpath)
starmask = hdulist[0].data
hdulist.close()

imshow(starmask)

#%%
# read header
filename = "../Images/A1_mosaic.fits" # with frame but no star
hdulist=fits.open(filename)
original = hdulist[0].data
header = hdulist[0].header
hdulist.close()


#%%
# load original image
filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()

#import convolution kernels
from kernels import convolve,EDGEDETECTION,GAUSSIANBLUR,gaussian


# Apply convolutions sequence of convolutions:
conv = convolve(image,gaussian(sigma=2,N=5))
conv = convolve(conv,EDGEDETECTION)

# plot results
plt.figure()
imshow(zscale(conv[440:490,1580:1620]))
plt.title('Convoluted image')

# create and save mask
threshold=15
mask=np.where(conv>threshold,1,0)
# hdu = fits.PrimaryHDU(mask) # save results
# hdu.writeto('mask.fits')
#%%

# =============================================================================
# Save original image
# =============================================================================

filename = "nice_images/galaxy_rejection_area.fits" # with frame but no star
hdulist=fits.open(filename)
effectivemask = hdulist[0].data
hdulist.close()

# plot and save whole image
plt.figure(figsize=(4,6))
original2 = np.where(original==0,1,original)
imshow(zscale(np.log(original2)))
imshow(np.where(effectivemask==0,1,np.nan), cmap = 'cool', alpha = 0.4)

ax = plt.gca()
ax.set_xlabel('$x$ axis [pixel unit]')
ax.set_ylabel('$y$ axis [pixel unit]')
plt.tight_layout()
plt.savefig('nice_images/originalimage.png', dpi = 400)


#%%

# =============================================================================
# Plot some of the images
# =============================================================================

up, down, left, right = 1200, 1500, 1700, 2000

# original
plt.figure(figsize=(6,6))
imshow(zscale(image[up:down,left:right]), extent = (left, right, down, up))
ax = plt.gca()
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('top') # the rest is the same
ax.set_xlabel(r'$x$ axis - [$\mathtt{pixel\ unit}$]',labelpad=10)
ax.set_ylabel(r'$y$ axis - [$\mathtt{pixel\ unit}$]')
plt.savefig('nice_images/original.png',dpi=400)

# convolution
plt.figure(figsize=(6,6))
im=imshow(zscale(conv[up:down,left:right]), extent=(left, right, down, up), cmap='coolwarm')
plt.colorbar(im,orientation='horizontal',fraction=0.040)#, pad=0.04)
ax = plt.gca()
# plt.colorbar(im,fraction=0.046, pad=0.04)
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('top') # the rest is the same
ax.set_xlabel(r'$x$ axis - [$\mathtt{pixel\ unit}$]',labelpad=10)
ax.set_ylabel(r'$y$ axis - [$\mathtt{pixel\ unit}$]')
plt.savefig('nice_images/convolution.png',dpi=400)

# mask
threshold=15
mask=np.where(conv>threshold,1,0)
plt.figure(figsize=(6,6))
imshow(mask[up:down,left:right], extent = (left, right, down, up))
ax = plt.gca()
ax.xaxis.set_label_position('top') 
ax.xaxis.set_ticks_position('top') # the rest is the same
ax.set_xlabel(r'$x$ axis - [$\mathtt{pixel\ unit}$]',labelpad=10)
ax.set_ylabel(r'$y$ axis - [$\mathtt{pixel\ unit}$]')
plt.savefig('nice_images/mask.png',dpi=400)
