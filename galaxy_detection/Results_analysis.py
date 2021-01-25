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

# print(f'{len(np.argwhere(galaxy_counts<0))}')

for gal_count in galaxy_counts:
    if gal_count > 0:
        galaxy_magnitudes.append(header["MAGZPT"] - 2.5 * np.log10(gal_count))

galaxy_magnitudes=np.array(galaxy_magnitudes)
print(f'max magn: {galaxy_magnitudes.max()}\nmin magn: {galaxy_magnitudes.min()}')




#%%
bins=np.arange(10,18,0.3)
number_galaxies=[]
for b in bins:
    N = len(np.argwhere(galaxy_magnitudes<b))
    number_galaxies.append(N)

logN=np.log10(number_galaxies)



# fitting line
fitting_range = (11,16)
goodindexes=np.argwhere((bins>=fitting_range[0]) & (bins<=fitting_range[1]))
fit,cov_matr = np.polyfit(bins[goodindexes][:,0],logN[goodindexes][:,0],1,cov=True)
polynomial = np.poly1d(fit)
print('Fit: ',fit)

plt.plot(bins,logN,marker='s', label='Collected data',linestyle='None',
         markersize=4,fillstyle='none')

plt.axvspan(fitting_range[0],fitting_range[1],alpha=0.1, label='fitting range')


plotrange=np.linspace(10,16,100)
plt.plot(plotrange,polynomial(plotrange),color='Black',linestyle= '-',linewidth=0.7,
         label='Linear fit')
plt.text(15,1.5,f'Gradient: {fit[0]:.2f}',fontsize=14)
plt.legend()
ax=plt.gca()
ax.set_xlabel(r'Magnitude $\mathrm{m}$')
ax.set_ylabel(r'$\mathrm{Log_{10}}$ (Number galaxies)')
plt.savefig("galaxy_brightness_analysis_results/Histogram_Numbers_magnitude.png")


plt.show()

#print(header["MAGZPT"])
#%%

    
zscale=visualization.ZScaleInterval()

data = np.loadtxt('galaxy_brightness_analysis_results/brightness_data.txt')

filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
header = hdulist[0].header
hdulist.close()

filename = "galaxy_brightness_analysis_results/background_image.fits" # with frame but no star
hdulist=fits.open(filename)
background_image = hdulist[0].data
header = hdulist[0].header
hdulist.close()


galaxylist = np.loadtxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt',skiprows = 1)

galaxy_flux = (data[:,0] - data[:,2]) * data [:,4]
galaxy_magnitudes = []
#%%
print(f'{len(np.argwhere(galaxy_flux<0))}')

negativeflux = np.argwhere(galaxy_flux<0)[:,0]
print(negativeflux [:20])

np.random.seed(1)
np.random.shuffle(negativeflux)

def show_negative_flux(index):
    
    row,col,max_value,num_pix = galaxylist[negativeflux[index]]
    row = int(row); col = int(col)

    print(f'\ncoordinates: \nrow = {row}\ncolumn = {col}')
    print(f'max value = {max_value}\nnumber pixels: {num_pix}')

    radius_outer = 30
    radius_inner = 15

    x = np.arange(2*radius_outer + 1) - radius_outer
    y = np.arange(2*radius_outer + 1) - radius_outer
    
    xx,yy=np.meshgrid(x,y)
    
    base_inner = ( yy**2 + xx**2 <= radius_inner**2 )
    base_outer = ( yy**2 + xx**2 <= radius_outer**2 ) * ( yy**2 + xx**2 > radius_inner**2 )
    
    # take a patch from the immage surrounding the galaxy
    crop_image = image[row - radius_outer : row + radius_outer + 1,\
                      col - radius_outer : col + radius_outer + 1]
    
    # take the corresponding patch frmo the background
    crop_back = background_image[row - radius_outer : row + radius_outer + 1,\
                      col - radius_outer : col + radius_outer + 1]
    
    
    background = crop_back * base_outer
    
    # crop_image =  crop_image * base_inner

    plt.subplot(1,2,1)
    imshow(zscale(crop_image))
    plt.subplot(1,2,2)
    imshow(background)
    
    plt.figure()
    imshow(zscale(image))
    plt.xlim(col - 60, col + 60)
    plt.ylim(row + 60, row - 60)
    plt.show()

show_negative_flux(3)

plt.figure()
imshow(zscale(image))

#%%

# select sky patch

rowrange = (3100,3500)
colrange = (500,1000)














