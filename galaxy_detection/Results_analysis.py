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

data = np.loadtxt('galaxy_brightness_analysis_results/brightness_data.txt')

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
