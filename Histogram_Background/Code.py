# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:35:37 2021

@author: martin
"""

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.stats import norm
#%%

#hdulist = fits.open("C:/MARTIN/Uni/Labs/Astro_Image/A1_mosaic.fits")
hdulist = fits.open("C:/MARTIN/Uni/Labs/Astro_Image/A1_mosaic_noframe.fits")

image = hdulist[0].data
hdulist.close()
print("Shape : f{image.shape}, original is 4611, 2570")


#%%

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

plt.plot(intensity, norm.pdf( intensity, mean, std))

plt.show()


#%%

plt.figure(figsize = [25, 15])
plt.imshow(image[::-1], vmin = 40000)#, cmap='gray')
plt.colorbar()
plt.show()


#%%

plt.imshow(image[::-1], cmap='gray')
plt.colorbar()
plt.show()