import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import tifffile as tiff
import os

#get path
path='Images/nostar_mask.tif'
filepath,filename=os.path.split(path)
filepath+='/'

# load tiff file
im=tiff.imread(filepath+filename)
print(im.shape)
plt.imshow(im)
#%%
# convert into fits and save
savename=filename[:-4]+'.fits'
hdu = fits.PrimaryHDU(im)
hdu.writeto(filepath+savename)    


