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
from scipy.optimize import curve_fit
from numpy.random import normal
plt.style.use('mystyle-2.mplstyle')


#%%


zscale=visualization.ZScaleInterval()

data = np.loadtxt('galaxy_brightness_analysis_results/brightness_data_2.txt')

filename = "../Images/A1_mosaic.fits" # with frame but no star
#filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
header = hdulist[0].header
hdulist.close()

filename = "galaxy_brightness_analysis_results/background_image.fits" # with frame but no star
hdulist=fits.open(filename)
background_image = hdulist[0].data
#header = hdulist[0].header
hdulist.close()


galaxylist = np.loadtxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt',skiprows = 1)


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
    imshow(zscale(background))
    
    plt.figure()
    imshow(zscale(image))
    plt.xlim(col - radius_outer, col + radius_outer)
    plt.ylim(row + radius_outer, row - radius_outer)
    plt.show()



def linear_function(x, a, b):
    return a * x + b


#%%


plt.plot(data[:,1], label = "mean background")
plt.plot(data[:,2], label = "background std")
plt.legend()
plt.show()


#%%


truncated_data = data[data[:,2]<100]

plt.plot(truncated_data[:,1], label = "mean background")
plt.plot(truncated_data[:,2], label = "mean background std")
plt.legend()
plt.show()



#%%

galaxy_counts = (data[:,0] - data[:,1]) * data [:,3]
galaxy_counts_error = ( data[:,0]  * data [:,3] )**0.5
galaxy_counts_error_back = data[:,2] * (data [:,3]**0.5)
galaxy_mag = []
galaxy_mag_error = []
galaxy_mag_error_back = []


# print(f'{len(np.argwhere(galaxy_counts<0))}')

for i in range(len(galaxy_counts)):
    if galaxy_counts[i] > 0 and 2.5 * np.log(10) * galaxy_counts_error[i] / galaxy_counts[i] <= 100:
        galaxy_mag.append(header["MAGZPT"] - 2.5 * np.log10(galaxy_counts[i]))
        galaxy_mag_error.append(2.5 * np.log(10) * galaxy_counts_error[i] / galaxy_counts[i] )
        galaxy_mag_error_back.append(2.5 * np.log(10) * galaxy_counts_error_back[i] / galaxy_counts[i])


galaxy_mag = np.array(galaxy_mag)
galaxy_mag_error = ( np.array(galaxy_mag_error)**2 + np.array(galaxy_mag_error_back)**2 + header["MAGZRR"]**2 )**0.5
print(f'max magn: {galaxy_mag.max()}\nmin magn: {galaxy_mag.min()}')

#%%


plt.plot(galaxy_mag_error, label = "intensity")
plt.plot( galaxy_mag_error_back , label = "back")
plt.legend()
plt.show()


#%%

plt.plot(galaxy_mag, label = "magnitude")
plt.plot(galaxy_mag_error, label = "magnitude error")
plt.legend()
plt.show()
#%%

mean = np.array([1,2,3])
std = np.array([3,2,1])

buf = normal(loc = mean, scale = std, size = (100000,3))

print(buf.mean(axis = 0))


print(buf < 2)


#%%
bins=np.arange(11,18,0.3)
number_galaxies = []
number_galaxies_error = []

repetitions = 100000

dist = normal(loc = galaxy_mag, scale = galaxy_mag_error/2, size = (repetitions, len(galaxy_mag)))

mean = dist.mean(axis = 0)
std  = dist.std(axis = 0) 

print(dist.shape, mean - galaxy_mag, std - galaxy_mag_error)

bins_N = np.zeros((len(bins), repetitions))

for j in range(repetitions):

    for i in range(len(bins)):
        
        bins_N[i,j] = len(np.argwhere(dist[j] < bins[i]))

number_galaxies = bins_N.mean(axis = 1)
number_galaxies_error = bins_N.std(axis = 1)

print("All objects are calculated : ", number_galaxies.sum(), "and are : ",  len(galaxy_mag))
    
number_galaxies_old = []

for i in range(len(bins)):
    number_galaxies_old.append(len(np.argwhere(galaxy_mag < bins[i])))

number_galaxies_old = np.array(number_galaxies_old)



#%%

plt.plot(number_galaxies, "rd",label = "new")
plt.plot(number_galaxies_error,label = "new error")
plt.plot(number_galaxies_old,  "bd", label = "old")
plt.legend()
plt.show()

#%%

bin_index = 15
plt.hist(bins_N[bin_index,:], bins = 86)
plt.show()

plt.figure()
plt.hist(np.log10(bins_N[bin_index,:]), bins = 86)
plt.show()

print(f" log mean : {np.log10(bins_N[bin_index,:]).mean()}, mean log : {np.log10(bins_N[bin_index,:].mean())}, mean : {np.log10(number_galaxies_old[bin_index])} ")


#%%
    
cov_in = np.zeros((len(number_galaxies), len(number_galaxies)))

    

logN = np.log10(number_galaxies)
logN_error = number_galaxies_error/number_galaxies

plt.plot(logN, label = "mean")
plt.plot(logN_error, label = "std")
plt.legend()
plt.show()

#%%


# fitting line
fitting_range = (11,16)
goodindexes=np.argwhere((bins>=fitting_range[0]) & (bins<=fitting_range[1]))

print( bins[goodindexes].shape, logN[goodindexes].shape, logN_error[goodindexes].shape)

fit,cov_matr = curve_fit(linear_function, bins[goodindexes].ravel(), logN[goodindexes].ravel(), sigma = logN_error[goodindexes].ravel())

#fit,cov_matr = np.polyfit(bins[goodindexes][:,0],logN[goodindexes][:,0],1,cov=True)
#polynomial = np.poly1d(fit)

print(f'Fit: ,{fit}, {cov_matr**0.5}')

plt.plot(bins,logN,marker='s', label='Collected data',linestyle='None',
         markersize=4,fillstyle='none')

plt.axvspan(fitting_range[0],fitting_range[1],alpha=0.1)


plotrange=np.linspace(10,16,100)
plt.plot(plotrange,linear_function(plotrange, fit[0], fit[1]),color='Black',linestyle= '-',linewidth=0.7,
         label='Linear fit')
plt.text(15,1.5,f'Gradient: {fit[0]:.2f}',fontsize=14)
plt.legend()
ax=plt.gca()
ax.set_xlabel(r'Magnitude $\mathrm{m}$')
ax.set_ylabel(r'$\mathrm{Log_{10}}$ (Number galaxies)')
# plt.savefig("galaxy_brightness_analysis_results/Histogram_Numbers_magnitude.png")


plt.show()




#%%


"""
print(f'{len(np.argwhere(galaxy_flux<0))}')

negativeflux = np.argwhere(galaxy_flux<0)[:,0]
print(negativeflux [:20])

np.random.seed(1)
np.random.shuffle(negativeflux)



show_negative_flux(3)

plt.figure()
imshow(zscale(image))

"""










