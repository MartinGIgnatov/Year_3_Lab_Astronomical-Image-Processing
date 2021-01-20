import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from astropy.io import fits
from PIL import Image
from kernels import convolve,EDGEDETECTION,GAUSSIANBLUR,gaussian
import astropy.visualization as visualization
plt.style.use('mystyle-2.mplstyle')
#%%
#import original image
filename = "galaxy_detection/A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()

# imshow(image)

# try applying sequence of convolutions:

conv = convolve(image,gaussian(sigma=2,N=5))
conv=convolve(conv,EDGEDETECTION)
# conv=np.where(conv<0,0,conv)

zscale=visualization.ZScaleInterval()
# imshow(zscale(conv)) # show convolved image
# imshow(zscale(conv[1200:1500,1800:2100]))


plt.figure()
imshow((conv[1200:1500,1700:2000]))
plt.title('Convoluted image')
plt.savefig('galaxy_detection/convoluted.png',dpi=400)



plt.figure()
imshow(zscale(image[1200:1500,1700:2000]))
plt.title('zscale(original)')
plt.savefig('galaxy_detection/original.png',dpi=400)

# plt.xlim(1000,2000)
# plt.ylim(2000,1000)
threshold=15
mask=np.where(conv[1200:1500,1700:2000]>threshold,1,0)
plt.figure()
imshow(mask)
plt.title(f'Mask, threshold: {threshold}')
plt.savefig('galaxy_detection/mask.png',dpi=400)

# save mask
mask=np.where(conv>threshold,1,0)
hdu = fits.PrimaryHDU(mask)
hdu.writeto('galaxy_detection/mask.fits')


# IDEA:
# can threshold values above 100 and then for those who got into the mask,
# move with a more gentle threshold (e.g. 60) not to miss points

#%%
# =============================================================================
# Try experimenting with astropy visualisation to get decent visualisations on python too!
# =============================================================================
mask=np.where(conv>15,1,0)

zscale=visualization.ZScaleInterval()
scaledimg=zscale(image)
zscaleconv=zscale(conv)


# xlim=(600,800)
# ylim=(2400,2200)
xlim=(1800,2100)
ylim=(1500,1200)

plt.figure()
imshow(scaledimg)
plt.xlim(*xlim)
plt.ylim(*ylim)

plt.figure()
imshow(zscaleconv)
plt.xlim(*xlim)
plt.ylim(*ylim)

plt.figure()
imshow(mask)
plt.xlim(*xlim)
plt.ylim(*ylim)

#%%
# =============================================================================
# Try experimenting with other operators
# =============================================================================
from kernels import convolve, GAUSSIANBLUR, gaussian, SOBELY, SOBELX

def show(img):
    plt.figure()
    imshow(img)
    plt.show()

zscale=visualization.ZScaleInterval()

patch = image[1200:1500,1700:2000]
blurred=convolve(patch, gaussian(sigma=2,N=5))

show(blurred)

sobelx=convolve(blurred,SOBELX)

show(sobelx)

# %%
