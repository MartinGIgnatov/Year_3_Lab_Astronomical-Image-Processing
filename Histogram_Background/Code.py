# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:35:37 2021

@author: martin
"""

import numpy as np
import os
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy as sp
import scipy.optimize
plt.style.use('../galaxy_detection/mystyle-2.mplstyle')
#%%

#hdulist = fits.open("/A1_mosaic.fits")
fileDir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join(fileDir, '../Images/A1_mosaic_noframe.fits')
hdulist = fits.open(filename)

image = hdulist[0].data
hdulist.close()
print(f"Shape : {image.shape}, original is 4611, 2570")


#%%

## CREATE A HISTOGRAM FOR THE IMAGE

data = image.flatten()

data = data[np.where(data>4000)]

data = data[data.nonzero()]  # Removes 0

print(f"Mean value is : {data.flatten().mean()}")
print(f"STD value is : {data.flatten().std()}")
print(f"Min  value is : {data.flatten().min()}")
print(f"Max  value is : {data.flatten().max()}")
plt.hist(data, bins = 500 )#, range = (3300,3700))
plt.show()

#%%

## EVALUATE BACKGROUND NOISE 

data = image.flatten()

up = 3486
low = 3350

data = data[ np.where( data > low)]
data = data[ np.where( data < up)]

data = data[data.nonzero()]  # Removes 0

mean, std = norm.fit(data)

print(f"Mean is : {mean}\nSTD  is : {std}")

n,bins,_ = plt.hist(data, bins = 135 , range = (low,up), density = True)

intensity = (bins[:-1] + bins[1:])/2

plt.plot(intensity, 1.1*norm.pdf( intensity, mean, std))

plt.show()


#%%
# =============================================================================
# Gaussian fit: sorry, i don't know how to use 
# =============================================================================
def gaussianfit(intensity, amp, mu, sigma):
    y = amp/(np.sqrt(2*np.pi)* sigma) * np.exp(-(intensity - mu)**2/(2*sigma**2))
    return y


val,bins = np.histogram(data, bins = 135, range= (low, up))

binfit = (bins[:-1] + bins[1:])/2

p0 = [1e7, 3400, 30]
fit, cov_matr = sp.optimize.curve_fit(gaussianfit, binfit, val, p0 = p0)

# plt.plot(binfit, val)
plt.hist(data, bins = 135+100, range= (low, up+100),alpha=0.5, label ='data')
plt.plot(binfit, gaussianfit(binfit, *fit),color ='Black', linestyle = '--',
            label = 'Gaussian Fit')
plt.legend()
plt.text(3500, 2.5e5, rf'$\mu =\ \ {fit[1]:.1f}$',fontsize = 14)
plt.text(3500, 2.2e5, rf'$\sigma =\ \ {fit[2]:.1f}$',fontsize = 14)
ax = plt.gca()
ax.set_xlabel('Pixel count')
ax.set_ylabel('Occurrences')
# plt.savefig('histogram_fit.png', dpi = 400)

# from mpl_toolkits.axes_grid.inset_locator import inset_axes
# # this is an inset axes over the main axes
# inset_axes = inset_axes(ax, 
#                     width="30%", # width = 30% of parent_bbox
#                     height=1.0, # height : 1 inch
#                     loc=4)
# plt.hist(image.flatten(), bins = 500)

# # #plt.title('Probability')
# plt.xticks([])
# plt.yticks([])


#%%

# Show no borders

image[np.where(image == 0 )] = 50000

plt.imshow(image[::-1], cmap='gray')
plt.colorbar()
plt.show()

