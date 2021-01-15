# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:04:07 2021

@author: marti
"""


import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import norm
#%%

#hdulist = fits.open("/A1_mosaic.fits")
fileDir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join(fileDir, '../Images/A1_mosaic_noframe.fits')
hdulist = fits.open(filename)

image = hdulist[0].data
hdulist.close()
print(f"Shape : {image.shape}, original is 4611, 2570")

#%%

image[np.where(image <  35000 )] = 0
image[np.where(image >= 35000 )] = 50000

plt.imshow(image, origin = "lower", cmap='gray')
plt.colorbar()
plt.show()
