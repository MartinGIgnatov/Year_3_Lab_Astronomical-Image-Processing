import numpy as np
from scipy.special import erf
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

###  Get prerequisites, as data and functions
###  The developed image we are working with and the header from original
###  Get Z-scale, used in visualization for better representation 

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


def linear_function(x, a, b):
    return a * x + b


#%%

###   plot background and its means
plt.figure()
plt.plot(data[:,1], label = "mean background")
plt.plot(data[:,2], label = "background std")
plt.legend()
plt.show()


#%%

###   Get rid of data that is too hihg in error, it comes from too close objects, where the bacgraound is mostly cropped
###   and cant be determined

truncated_data = data[data[:,2]<100]

###  Show truncated background
plt.figure()
plt.plot(truncated_data[:,1], label = "mean background")
plt.plot(truncated_data[:,2], label = "mean background std")
plt.legend()
plt.show()



#%%


###   convert from counts to intensity and propagate the error from poisson on intensity counts,
###   background std and the normalization constants error


# data: # inner_mean, background_mean, backrgound_error, inner_number_pixels, outer_number_pixels
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
        galaxy_mag_error.append(2.5  * galaxy_counts_error[i] / galaxy_counts[i] / np.log(10) )
        galaxy_mag_error_back.append(2.5 * galaxy_counts_error_back[i] / galaxy_counts[i] / np.log(10))


galaxy_mag = np.array(galaxy_mag)
galaxy_mag_error = ( np.array(galaxy_mag_error)**2 + np.array(galaxy_mag_error_back)**2 + header["MAGZRR"]**2 )**0.5
print(f'max magn: {galaxy_mag.max()}\nmin magn: {galaxy_mag.min()}')


#%%

###   plot magnitudes and the two main sources of error

plt.figure()
plt.plot(galaxy_mag, label = "magnitude")
plt.plot(galaxy_mag_error, label = "intensity error")
plt.plot( galaxy_mag_error_back , label = "back error")
plt.legend()
plt.show()

print(f" The mag are between {min(galaxy_mag)} and {max(galaxy_mag)}")



#%%

###               SIMULATION    to convert each magnitudes error into bins error and find the mean bin size
###                             then convert to log of the counts
###                             Find the values of the bins with mean mags as the old bin values

bins=np.arange(11,18,0.3)
number_galaxies = []
number_galaxies_error = []

repetitions = 100000

dist = normal(loc = galaxy_mag, scale = galaxy_mag_error, size = (repetitions, len(galaxy_mag)))

mean = dist.mean(axis = 0)
std  = dist.std(axis = 0) 

print(dist.shape, mean - galaxy_mag, std - galaxy_mag_error)

bins_N = np.zeros((len(bins), repetitions))

for j in range(repetitions):
    #stdout.write(f'\r{j}/{repetitions}')
    #stdout.flush()
    for i in range(len(bins)):
        
        bins_N[i,j] = len(np.argwhere(dist[j] < bins[i]))

number_galaxies = bins_N.mean(axis = 1)
number_galaxies_error = bins_N.std(axis = 1)

    
number_galaxies_old = []

for i in range(len(bins)):
    number_galaxies_old.append(len(np.argwhere(galaxy_mag < bins[i])))

number_galaxies_old = np.array(number_galaxies_old)


#%%

###  plot the counts for with and without sims and the error from sims
plt.figure()
plt.plot(number_galaxies, "rd",label = "new")
plt.plot(number_galaxies_error,label = "new error")
plt.plot(number_galaxies_old,  "bd", label = "old")
plt.legend()
plt.show()

#%%

###   plot the distribution of counts for a specific bin and the dist for log of counts for the same bin

bin_index = 15
plt.figure()
plt.hist(bins_N[bin_index,:], bins = 86)
plt.show()

plt.figure()
plt.hist(np.log10(bins_N[bin_index,:]), bins = 86)
plt.show()

print(f" log mean : {np.log10(bins_N[bin_index,:]).mean()}, mean log : {np.log10(bins_N[bin_index,:].mean())}, mean : {np.log10(number_galaxies_old[bin_index])} ")


#%%
    
####       Calculate the two sources of error, from the poisson and the mags
    
####  INTENSITY AND POISSON ERROR
logN = np.log10(number_galaxies)
logN_error = number_galaxies_error/number_galaxies/np.log(10)

poisson_error = np.sqrt(number_galaxies)

logN_error_total = (logN_error**2 + poisson_error**2)**0.5

N_error = np.sqrt(number_galaxies_error**2 + poisson_error**2)
log_N_error_up = np.log10(number_galaxies + N_error) - logN
log_N_error_down = logN - np.log10(number_galaxies - N_error)

"""
####  ONLY POISSON ERROR
logN = np.log10(number_galaxies_old)
logN_error = 1/(number_galaxies**0.5)/np.log(10)
"""


#%% 

###    Compare two errors

relative_poisson = poisson_error/number_galaxies
relative_brightness = number_galaxies_error/number_galaxies

relative_poisson_brightness = poisson_error/number_galaxies_error

plt.figure()
ax=plt.gca()
line1 = ax.plot(bins, relative_poisson * 100,label=r'Poisson Percentage Error')
line2 = ax.plot(bins, relative_brightness * 100,label=r'Brightness Percentage Error',
         color = 'Black', linestyle='--')

# plot ratio of two errors
ax2colour = 'Blue'
ax.set_xlabel(r'Magnitude $\mathrm{m}$')
ax.set_ylabel(r'[Error $\div$ No. Galaxies] $\times 100$')
ax2 = ax.twinx()
line3 = ax2.plot(bins, relative_poisson_brightness, color=ax2colour,
                 label=r'Poisson error $\div$ Brightness errors')

# create common legend
lines = line1+line2#+line3
labs = [l.get_label() for l in lines]
ax2.legend(lines, labs, facecolor='lightskyblue',framealpha=0.5,loc = 1,fontsize=13)

ax2.tick_params(axis='y', colors=ax2colour)
ax2.set_ylabel(r'Poisson error $\div$ Brightness errors', rotation = -90,labelpad = 20,
               color=ax2colour)
ax2.set_ylim(0,5)
plt.savefig('galaxy_brightness_analysis_results/errors_comparison.png',dpi=400)




#%%

# fitting line
fitting_range = (11,16)
goodindexes=np.argwhere((bins>=fitting_range[0]) & (bins<=fitting_range[1]))

print( bins[goodindexes].shape, logN[goodindexes].shape, logN_error[goodindexes].shape)

fit,cov_matr = curve_fit(linear_function, bins[goodindexes].ravel(), logN[goodindexes].ravel(), sigma = logN_error[goodindexes].ravel())

#fit,cov_matr = np.polyfit(bins[goodindexes][:,0],logN[goodindexes][:,0],1,cov=True)
#polynomial = np.poly1d(fit)

print(f'Fit: ,{fit}, {cov_matr}')
plt.figure()
plt.errorbar(bins,logN,yerr = (log_N_error_up, log_N_error_down), marker='s',
             label='Resampled data',linestyle='None',
         markersize=4,fillstyle='none', capsize = 2, color = 'Black')

plt.plot(bins, np.log10(number_galaxies_old), marker = 'o', color ='Green', linestyle = 'None',
         fillstyle='none', label = 'Original Data')

plt.axvspan(fitting_range[0],fitting_range[1],alpha=0.1)


plotrange=np.linspace(11,16,100)
plt.plot(plotrange,linear_function(plotrange, fit[0], fit[1]),color='Black',linestyle= '-',linewidth=0.7,
         label='Linear fit')
plt.text(13,1.5,rf'Gradient: ${fit[0]:.3f}\pm{np.sqrt(cov_matr[0][0]):.3f}$',fontsize=14)
plt.legend()
ax=plt.gca()
ax.set_xlabel(r'Magnitude $\mathrm{m}$')
ax.set_ylabel(r'$\mathrm{Log_{10}}$ (Number galaxies)')
plt.savefig("galaxy_brightness_analysis_results/Histogram_Numbers_errorbars.png")


# plt.show()


from simulate_distribution import get_points
low = 0
high = 18

a = get_points(3837*100,0.36111444, -2.67273998,low,high)

binrange = np.arange(11,high,0.3)

totalN = []

for bn in binrange:
    totalN.append(len(np.argwhere(a <= bn)))
totalN = np.array(totalN)
print(f'total N: {totalN}')

skipfirst = 0
binrange = binrange[skipfirst:]
totalN = totalN[skipfirst:]

#%%

number_galaxies_old_separate = []

for i in range(len(bins) - 1):
    number_galaxies_old_separate.append(len(np.argwhere( np.abs(galaxy_mag - bins[i] - 0.15) < 0.15 )))

number_galaxies_old_separate = np.array(number_galaxies_old_separate)[goodindexes]

plt.plot(number_galaxies_old_separate)


#%%

###   Generate fake points

def error_func(x, a, b ,d):
    return np.exp(x/b - a) + d

fit_galaxy_mag = galaxy_mag[galaxy_mag<18]
fit_galaxy_mag_error = galaxy_mag_error[galaxy_mag<18]

plt.figure()
plt.plot(fit_galaxy_mag, fit_galaxy_mag_error,"rd")

fit_fake,_ = curve_fit(error_func, fit_galaxy_mag.ravel(), fit_galaxy_mag_error.ravel())#, p0 = )

x = np.arange(0,20,0.01)

plt.plot(x, error_func(x, fit_fake[0], fit_fake[1], fit_fake[2]))


print(fit_fake)


N_below = int(10**(fit[0] * 10.7 + fit[1]) - 10**(fit[1]))
N_above = int(10**(fit[0] * 24 + fit[1]) - 10**(fit[0] * 16.3 + fit[1]))

fake_points = np.concatenate((get_points( N_below, fit[0], fit[1], 0, 10.7), get_points( N_above, fit[0], fit[1], 16.3, 24)))

fake_points = get_points( N_below, fit[0], fit[1], 0, 10.7)

fake_points_errors = error_func(fake_points, fit_fake[0], fit_fake[1], fit_fake[2])

fake_real_mag =  np.concatenate((galaxy_mag[galaxy_mag < 18], fake_points  ))
fake_real_mag_error =  np.concatenate((galaxy_mag_error[galaxy_mag < 18], fake_points_errors  ))

#%%

plt.figure()
plt.hist(fake_real_mag, bins = 1000)
plt.show()


#%%

repetitions_real_fake = int(repetitions/10)

dist = normal(loc = fake_real_mag, scale = fake_real_mag_error, size = (repetitions_real_fake, len(fake_real_mag)))

mean = dist.mean(axis = 0)
std  = dist.std(axis = 0) 

print(dist.shape, mean - fake_real_mag, std - fake_real_mag_error)

bins_N_real_fake = np.zeros((len(bins), repetitions_real_fake))


for j in range(repetitions_real_fake):
    
    for i in range(len(bins)):
        
        bins_N_real_fake[i,j] = len(np.argwhere(dist[j] < bins[i]))

number_galaxies_real_fake = bins_N_real_fake.mean(axis = 1)
number_galaxies_real_fake_error = bins_N_real_fake.std(axis = 1)


logN_real_fake = np.log10(number_galaxies_real_fake)
logN_real_fake_error = number_galaxies_real_fake_error/number_galaxies_real_fake/np.log(10)

bins_N_real_fake_basic = []
for i in range(len(bins)):
        
    bins_N_real_fake_basic.append( len(np.argwhere(fake_real_mag < bins[i])))


plt.figure()
plt.plot(bins,logN_real_fake, label = "real_fake")
plt.plot(bins, np.log10(bins_N_real_fake_basic), label = "Basic fake real")
plt.plot(bins,logN, label = "sim")
plt.plot(bins,np.log10(number_galaxies_old), label = "old")
plt.legend()
plt.show()







#%%
"""
### calculate and plot the expected number of shifts per bin to justify 
### WRONG
shift_expectation = []

for i in range(1, len(bins)):
    below_indeces = np.argwhere(galaxy_mag < bins[i])
    above_indeces = np.argwhere(galaxy_mag > bins[i])
    
    up_shift = np.sum( 0.5 - 0.5 * erf((bins[i] - galaxy_mag[below_indeces])/galaxy_mag_error[below_indeces] )/galaxy_mag_error[below_indeces]  )
    
    down_shift = np.sum( 0.5 - 0.5 * erf( (galaxy_mag[above_indeces] - bins[i])/galaxy_mag_error[above_indeces] )/galaxy_mag_error[above_indeces]  )
    
    shift_expectation.append(down_shift - up_shift)
    
plt.figure()
plt.plot(bins[:-1], shift_expectation, "r--", label = "Expected number increases")
plt.hlines(0, xmin = min(bins[:-1]), xmax = max(bins[:-1]), label = "0th line")
plt.legend()
plt.show()


#%%

### Old material for data anakysis
"""
"""

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



print(f'{len(np.argwhere(galaxy_flux<0))}')

negativeflux = np.argwhere(galaxy_flux<0)[:,0]
print(negativeflux [:20])

np.random.seed(1)
np.random.shuffle(negativeflux)



show_negative_flux(3)

plt.figure()
imshow(zscale(image))



#%%

###       TEST  for  normal, does do anything for the rest

mean = np.array([1,2,3])
std = np.array([3,2,1])

buf = normal(loc = mean, scale = std, size = (100000,3))

print(buf.mean(axis = 0))


print(buf < 2)

"""











