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

filename = "../Images/A1_mosaic.fits" # with frame but no star
hdulist=fits.open(filename)
header = hdulist[0].header
hdulist.close()



galaxy_counts = (data[:,0] - data[:,2]) * data [:,4]
galaxy_magnitudes = []

print(f'{len(np.argwhere(galaxy_counts<0))}')

for gal_count in galaxy_counts:
    if gal_count > 0:
        galaxy_magnitudes.append(header["MAGZPT"] - 2.5 * np.log10(gal_count))

galaxy_magnitudes=np.array(galaxy_magnitudes)
print(f'max magn: {galaxy_magnitudes.max()}\nmin magn: {galaxy_magnitudes.min()}')
#%%
bins=np.arange(10,25,0.3)
number_galaxies=[]
for b in bins:
    N = len(np.argwhere(galaxy_magnitudes<b))
    number_galaxies.append(N)

logN=np.log10(number_galaxies)



# fitting line
fitting_range = (9,16)
goodindexes=np.argwhere((bins>=fitting_range[0]) & (bins<=fitting_range[1]))
fit,cov_matr = np.polyfit(bins[goodindexes][:,0],logN[goodindexes][:,0],1,cov=True)
polynomial = np.poly1d(fit)
print('Fit: ',fit)

plt.plot(bins,logN,marker='d', label='collected data',linestyle='None',markersize=4)

plotrange=np.linspace(10,16,100)
plt.plot(plotrange,polynomial(plotrange),color='Red',linestyle= '--')

plt.legend()
ax=plt.gca()
ax.set_xlabel('magnitude m')
ax.set_ylabel(r'$\mathrm{Log_{10}}$ (Number galaxies)')
plt.savefig("galaxy_brightness_analysis_results/Histogram_Numbers_magnitude.png")


plt.show()

#print(header["MAGZPT"])
