import numpy as np
from scipy.special import erf
import scipy as sp
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
### Old material for data anakysis
galaxylist = np.loadtxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt')

filename = "A1_mosaic_nostar.fits" # with frame but no star
#filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()


filename = "fake_images/background_image.fits" # with frame but no star
hdulist=fits.open(filename)
background_image = hdulist[0].data
#header = hdulist[0].header
hdulist.close()


radius_list = np.loadtxt('fake_images/radius_list.txt').astype(int)

imshow(image)


filename = "../Images/A1_mosaic.fits" # with frame but no star
#filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
header = hdulist[0].header
hdulist.close()

zscale = visualization.ZScaleInterval()
#%%

def show_galaxy(index):
    
    row,col,max_value,num_pix = galaxylist[index]
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
    
    # take the corresponding patch from the background
    crop_back = background_image[row - radius_outer : row + radius_outer + 1,\
                      col - radius_outer : col + radius_outer + 1]
    
    
    background = crop_back * base_outer
    
    # crop_image =  crop_image * base_inner

    plt.subplot(1,2,1)
    imshow(zscale(crop_image))
    plt.subplot(1,2,2)
    imshow(zscale(background))
    
    plt.figure()
    imshow(image[row - radius_outer:row + radius_outer,col - radius_outer:col + radius_outer])
    # plt.xlim(col - radius_outer, col + radius_outer)
    # plt.ylim(row + radius_outer, row - radius_outer)
    plt.show()


def get_circular_kernel(radius, halfside):
    x = np.arange(2*halfside + 1) - halfside
    y = np.arange(2*halfside + 1) - halfside
    
    xx,yy=np.meshgrid(x,y)
    
    base = ( yy**2 + xx**2 <= radius**2)
    
    return base.astype('int')

def galaxy_brightness(row, col, image, background_image, radius):
    # if image_patch.size[0]< 36:
        # raise Exception('The patch should be larger than 36 x 36 !')
    shape = image.shape
    row, col = int(row), int(col)    
    
    inner_radius = radius
    outer_radius = np.ceil(inner_radius * 5 / 4).astype('int')
    
    halfside = 36 # maximum outer radius

        
    patch = image[row - halfside : row + halfside + 1,\
                      col - halfside : col + halfside + 1]
    
    patch_background = background_image[row - halfside : row + halfside + 1,\
                      col - halfside : col + halfside + 1]
    
    
    x = np.arange(2*halfside + 1) - halfside
    y = np.arange(2*halfside + 1) - halfside
    
    xx,yy=np.meshgrid(x,y)
    
    
    inner_circle = get_circular_kernel(inner_radius,halfside)
    
    
    outer_circle = get_circular_kernel(outer_radius,halfside)
    outer_crown = np.absolute(outer_circle - inner_circle)

    innerpatch = (patch * inner_circle)
    inner_values = innerpatch[innerpatch.nonzero()]
    
    outerpatch = (patch_background * outer_crown)
    outer_values = outerpatch[outerpatch.nonzero()]
    
    plt.figure()
    
    # plt.figure()
    imshow(patch_background)
    imshow(outer_crown, cmap = 'cool',alpha = 0.5)

    
    
    # inner_values_cleaned = inner_values - background_mean
    # brightness = np.sum(inner_values_cleaned)

    #record  data
    background_mean = np.mean(outer_values)
    background_std = np.std(outer_values)
    inner_mean = np.mean(inner_values)
    number_pixels_inner = len(inner_values)
    number_pixels_outer = len(outer_values)
    
    # innersum = np.sum(inner_values)
    # poisson_error_inner = np.sqrt(innersum) # adding poisson in quadrature
                    
    # brightness = innersum - number_pixels * background_mean
    # std_error_background = number_pixels * np.std(outer_values)

            
    # total_error = np.sqrt(poisson_error_inner**2 + 15**2)
            
                    
    return inner_mean, background_mean, background_std, number_pixels_inner, number_pixels_outer



def flux(index):
    galaxy = galaxylist[index]
    print(f'radius: {radius_list[index]}')
    info = galaxy_brightness(galaxy[0],galaxy[1], image, background_image,
                             radius_list[index])
    
    inner_mean, background_mean, backgroudn_std, pixels_in, pixels_out = info
    print(f'\n\ninner mean: {inner_mean}')
    print(f'bk mean: {background_mean}')
    print(f'inner pixels: {pixels_in}')
    flux_image = (inner_mean - background_mean) * pixels_in
    
    print(f'flux of galaxy: {flux_image:.2f}')
index = 280#812 #400
show_galaxy(index)
flux(index)

#%%
fluxlist = np.r_[3600, 1245, 3141, 3478, 2434, 7687, 5783, 6935, 1204, 1677, 4169]

diameter = np.r_[12, 10, 8, 10, 8, 10, 10, 12, 6, 8, 10]

radius = diameter/2

hist, bin_edges = np.histogram(radius)

bin_val = (bin_edges + (bin_edges[1]-bin_edges[0]))[:-1]
plt.plot(bin_val, hist, marker = 's', linestyle = 'None')

binfit = bin_val[hist!=0]
histfit = hist[hist!=0]


def gaussian_func(x, A, mu, sigma):
    y = A/(np.sqrt(2*np.pi) * sigma) * np.exp(-(x-mu)**2/(2*sigma**2))
    return y


p0 = [1, 5,2]
fit, cov_matr = sp.optimize.curve_fit(gaussian_func, binfit, histfit,p0=p0)
print(f'Fit: {fit}')
# plt.plot(diam, hist)

radius_range = np.linspace(3, 8, 100)

plt.figure()
plt.plot(radius_range, gaussian_func(radius_range,*fit))
plt.plot(binfit, histfit, marker = 'o')
#%%
plt.figure()
plt.scatter(diameter**2, fluxlist)
#%%

average_intensity = np.mean(fluxlist)

#%%

def gaussian_2D(x,y, amp, sigma):
    z = amp/(2*np.pi * sigma**2) * np.exp(-(x**2 + y**2)/(2*sigma**2))
    return z


def gaussian_blob(amp):
    
    sampled_radius = np.random.normal(fit[1],fit[2])
    # print(sampled_radius)
    sigma = sampled_radius * 0.4
    size = 128
    background = 3418
    bk_std = 12
    
    x = np.arange(size)-int(size/2)
    y = np.arange(size)-int(size/2)
    
    xx,yy=np.meshgrid(x,y)
    gauss = gaussian_2D(xx,yy, amp, sigma)
    background_noised = np.random.normal(background, bk_std, (size,size))
    gauss+=background_noised
    # gauss = np.random.poisson(gauss)
    # imshow(gauss)
    # plt.colorbar()
    return gauss

# gaussian_blob(50)

#%%
# define pipeline for spotting galaxy
from galaxy_list_filter import reject_galaxies_brightness
threshold=15


def convolution(image):
    '''
    Return threshold mask
    '''

    conv = convolve(image,gaussian(sigma=2,N=5))
    conv=convolve(conv,EDGEDETECTION)
    return np.where(conv>threshold,1, 0)


def index_galaxies(image,mask,framewidth = 150):
    """
    returns a list of galaxy location in the following format:
        
        row_number      col_number      max_brightnes       number_pixels
    
    where max_brightnes is the maximum value among the pixels in the cluster
    as checked in the original image

    Parameters
    ----------
    image : 2D matrix
        image containing the galaxies.
    mask : 2D matrix
        mask which is 1 where a galaxy was spotted and 0 otherwise.
    framewidth : int, optional
        Set a frame around the mask to zero. The default is 150.

    Returns
    -------
    galaxylist : (m x 4) matrix
        contains info about the galaxies: (column, row, max_pixel, number_pixels).

    """
    galaxylist=[]
    tempmask=mask.copy()
    
    # set frame around mask to zero
    tempmask[:framewidth,:]=0
    tempmask[-framewidth:,:]=0
    tempmask[:,:framewidth]=0
    tempmask[:,-framewidth:]=0
    
    i=1 # counting iterations
    while len(np.argwhere(tempmask==1))!=0:
        
        # for every white pixel in the mask, find its cluster
        
        # stdout.write(f"\rGalaxy number: {i} / {len(np.argwhere(tempmask==1))}      ")
        # stdout.flush()
        i+=1
        
        whitepixelcoor_y,whitepixelcoor_x=np.argwhere(tempmask==1)[0]
        tempmask[whitepixelcoor_y,whitepixelcoor_x]=0 # remove pixel from the mask
        checklist=[np.r_[whitepixelcoor_y,whitepixelcoor_x]] # add pixel to checklist
        white_pixels=[(whitepixelcoor_y,whitepixelcoor_x)] # add pixel to white pixels list
        
        # defining the functions before the loop should make the program faster
        # checklistappend=checklist.append
        checklistpop=checklist.pop
        # whitepixelsappend=white_pixels.append
        whitepixelextend=white_pixels.extend
        checklistextend=checklist.extend
        
        while len(checklist) != 0:
            # look for neighbouring elements  of pixels in checklist
            checkpixel=checklistpop(0)
            localmask = tempmask[checkpixel[0]-2:checkpixel[0]+3,checkpixel[1]-2:checkpixel[1]+3]
            localwhitepixels= checkpixel - 2 + np.argwhere(localmask==1) 
            whitepixelextend(localwhitepixels)
            checklistextend(localwhitepixels)
            tempmask[checkpixel[0]-2:checkpixel[0]+3,checkpixel[1]-2:checkpixel[1]+3] = 0
        
        white_pixels=np.array(white_pixels) # list of white pixels in the cluster
        numberpixels=len(white_pixels)
        
        # get brightness of pixels in the cluster
        imagepoints=image[white_pixels[:,0],white_pixels[:,1]]
        brightestindex=np.argwhere(imagepoints==imagepoints.max())[0]
        brightestpixel=white_pixels[brightestindex][0]
        brightestvalue=image[brightestpixel[0],brightestpixel[1]]
        
        galaxylist.append((brightestpixel[0], brightestpixel[1], brightestvalue, numberpixels))
    return galaxylist



image = gaussian_blob(5000)
mask = convolution(image)
plt.figure()
imshow(mask)
plt.figure()
imshow(image)
galaxylist = np.r_[index_galaxies(image, mask, 1)]
galaxylist = reject_galaxies_brightness(galaxylist, min_brightness=3465,
                                        max_brightness=35000)
print('found galaxies: ', len(galaxylist))

# ,min_brightness=3465,
                       
magnitude_list = np.arange(11,22,0.2)
counts_list = 10**((header["MAGZPT"] - magnitude_list)/2.5)                       # max_brightness=35000
# counts_list = np.linspace(10, 80000,40) # list of counts (intensities)
# magnitude_list = header["MAGZPT"] - 2.5 * np.log10(counts_list)
print(f'magnitude list: {magnitude_list}')

repeat = 500 # number of measurements per count value
ratio_value = np.zeros(counts_list.shape)
for j,c in enumerate(counts_list):
    print(j)
    for i in range(repeat):
        image = gaussian_blob(c)
        mask = convolution(image)
        # plt.figure()
        # imshow(mask)
        # plt.figure()
        # imshow(image)
        galaxylist = np.r_[index_galaxies(image, mask, 1)]
        galaxylist = reject_galaxies_brightness(galaxylist, min_brightness=3465,
                                                max_brightness=35000)
    
        if len(galaxylist) >= 1 and len(galaxylist)<=2:
            ratio_value[j]+=1
 #%%
print(ratio_value)
plt.plot(magnitude_list, ratio_value/repeat)

extintion = ratio_value/repeat

ax = plt.gca()

ax.set_ylabel('Extintion ratio')
ax.set_xlabel('Magnitude m')
np.savetxt('galaxy_brightness_analysis_results/extintion_ratio.txt', np.c_[magnitude_list, extintion],
            header = 'magnitude, \tratio')

