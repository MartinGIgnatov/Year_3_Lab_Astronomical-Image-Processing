import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from astropy.io import fits
import sys
from kernels import convolve,EDGEDETECTION,GAUSSIANBLUR,gaussian
import astropy.visualization as visualization
import time
from sys import stdout
import tifffile as tiff
import galaxy_list_filter as galaxyfilter
plt.style.use('mystyle-2.mplstyle')
zscale = visualization.ZScaleInterval()



def get_circular_kernel(radius, halfside):
    x = np.arange(2*halfside + 1) - halfside
    y = np.arange(2*halfside + 1) - halfside
    
    xx,yy=np.meshgrid(x,y)
    
    base = ( yy**2 + xx**2 <= radius**2)
    
    return base.astype('int')


def get_petrosian_radius(row, col, image,mask):
    # if image_patch.size[0]< 36:
        # raise Exception('The patch should be larger than 36 x 36 !')
    shape = image.shape
    row, col = int(row), int(col)
    print(f'row: {row}, col: {col}')
    
    radius_list = []
    brightness_list = []
    
    index = 0
    for inner_radius in range(2, 15, 1):
        outer_radius = np.ceil(inner_radius * 5 / 4).astype('int')
        
        halfside = 36 # maximum outer radius
    
            
        patch = image[row - halfside : row + halfside + 1,\
                          col - halfside : col + halfside + 1]
        
        patchmask = mask[row - halfside : row + halfside + 1,\
                          col - halfside : col + halfside + 1]
        
        
        x = np.arange(2*halfside + 1) - halfside
        y = np.arange(2*halfside + 1) - halfside
        
        xx,yy=np.meshgrid(x,y)
        
        
        inner_circle = get_circular_kernel(inner_radius,halfside)
        
        
        outer_circle = get_circular_kernel(outer_radius,halfside)
        outer_crown = np.absolute(outer_circle - inner_circle)
    
        innerpatch = (patch * inner_circle)
        inner_values = innerpatch[innerpatch.nonzero()]
        
        outerpatch = (patch * outer_crown)
        outer_values = outerpatch[outerpatch.nonzero()]
        
        background_mean = np.mean(outer_values)
        
        inner_values_cleaned = inner_values - background_mean
        brightness = np.sum(inner_values_cleaned)
        if index != 0:
            if brightness < brightness_list[-1]:
                #record proper data
                number_pixels = len(inner_values)
                innersum = np.sum(inner_values)
                poisson_error_inner = np.sqrt(innersum) # adding poisson in quadrature
                                
                brightness = innersum - number_pixels * background_mean
                std_error_background = number_pixels * np.std(outer_values)
                
                
                
                print('brightness: ', brightness)
                
                print('error poisson: ', poisson_error_inner)
                
                print('background mean: ', background_mean)
                
                print('error std: ', std_error_background)
                
                
                total_error = np.sqrt(poisson_error_inner**2 + std_error_background**2)
                break
                
                
        
        radius_list.append(inner_radius)
        brightness_list.append(brightness)
        index += 1
        
    return brightness, total_error
    # UNCOMMENT FOR PLOTTING
    # plt.figure()
    # imshow(inner_circle,alpha = 0.4)
    # imshow(outer_crown,alpha=0.4,cmap='seismic')
    # plt.axvline(halfside)
    # plt.axhline(halfside)
    
    # plt.figure()
    # imshow(mask)
    
    # plt.figure()
    # imshow(zscale(patch))
    # imshow(np.where(outer_crown!=0, outer_crown,np.nan),alpha=0.4,cmap='seismic_r')
    # ax = plt.gca()
    # ax.set_xlabel('x axis [pixel unit]')
    # ax.set_ylabel('y axis [pixel unit]')
    # ax.set_title('Outer Crown on Image')
    # # plt.savefig('nice_images/radius-nearby-galaxies.png',dpi = 400)
    
    # return radius_list, brightness_list
# img = np.ones((40, 40))
# get_petrosian_radius(300,300,image)

#%%
#Import actual image
filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()

#Import convolution threshold mask
filename = "mask.fits" # with frame but no star
hdulist=fits.open(filename)
mask = hdulist[0].data
hdulist.close()

# import galaxylist
galaxylist = np.loadtxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt')
# galaxylist = np.loadtxt('located_galaxies_00/galaxypositions-final.txt')

gal = galaxylist[2300] # 1000?
# 700 works well!
# now 870 seems to be working: smaller outer circle
# 380 nice proof of concept


x,y = get_petrosian_radius(gal[0], gal[1], image, mask)
print(x,y)
#%%
# =============================================================================
# remove background first
# =============================================================================

def delete_galaxy(row, col, image,background_image):
    # if image_patch.size[0]< 36:
        # raise Exception('The patch should be larger than 36 x 36 !')
    row, col = int(row), int(col)
    
    radius_list = []
    brightness_list = []
    
    index = 0
    for inner_radius in range(2, 15, 1):
        outer_radius = np.ceil(inner_radius * 5 / 4).astype('int')
        
        halfside = 36 # maximum outer radius
    
            
        patch = image[row - halfside : row + halfside + 1,\
                          col - halfside : col + halfside + 1]

        
        x = np.arange(2*halfside + 1) - halfside
        y = np.arange(2*halfside + 1) - halfside
        
        xx,yy=np.meshgrid(x,y)
        
        
        inner_circle = get_circular_kernel(inner_radius,halfside)
        
        
        outer_circle = get_circular_kernel(outer_radius,halfside)
        outer_crown = np.absolute(outer_circle - inner_circle)
    
        innerpatch = (patch * inner_circle)
        inner_values = innerpatch[innerpatch.nonzero()]
        
        outerpatch = (patch * outer_crown)
        outer_values = outerpatch[outerpatch.nonzero()]
        
        background_mean = np.mean(outer_values)
        
        inner_values_cleaned = inner_values - background_mean
        brightness = np.sum(inner_values_cleaned)
        if index != 0:
            if brightness < brightness_list[-1]:
                break
        
        radius_list.append(inner_radius)
        brightness_list.append(brightness)
        index += 1
    
    # print(inner_circle.shape)
    # plt.figure()
    # imshow(patch * (inner_circle ^ 1))
    # plt.title(f'radius: {inner_radius}')
    
    
    background_image[row - halfside : row + halfside + 1,\
                          col - halfside : col + halfside + 1] *=  (inner_circle ^ 1)
    # imshow(zscale(patch * (inner_circle ^ 1)))
    return inner_radius



background_image = image.copy()
galaxylist = np.loadtxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt')
radius_list = []
numgalaxies = len(galaxylist)
for index, gal in enumerate(galaxylist):
    sys.stdout.write(f'\rGalaxy {index:04d} / {numgalaxies}')
    sys.stdout.flush()
    radius = delete_galaxy(gal[0], gal[1], image, background_image)
    radius_list.append(radius)


plt.figure()
imshow(zscale(background_image))

hdu = fits.PrimaryHDU(background_image)
hdu.writeto('fake_images/background_image.fits')
#%%
np.savetxt('fake_images/radius_list.txt', radius_list)

#%%
# =============================================================================
# Calculate brightness
# =============================================================================

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

inner_mean_list = []
background_mean_list = []
backgroud_std_list = []
number_pixels_inner = []
number_pixels_outer = []

for index, gal in enumerate(galaxylist):
    sys.stdout.write(f'\rGalaxy {index:04d} / {numgalaxies}')
    sys.stdout.flush()
    rad = radius_list[index]
    output = galaxy_brightness(gal[0],gal[1],image,background_image,rad)
    if np.isnan(output[2]):
        continue
    inner_mean_list.append(output[0])
    background_mean_list.append(output[1])
    backgroud_std_list.append(output[2])
    number_pixels_inner.append(output[3])
    number_pixels_outer.append(output[4])
# #%%
# brightness_error = np.array(brightness_error)
# print(brightness_error.max())
# plt.hist(brightness_error, bins = 30)

#%%
# np.savetxt('galaxy_brightness_analysis_results/brightness_data_2.txt',
#            np.c_[inner_mean_list, background_mean_list, backgroud_std_list,
#                  number_pixels_inner, number_pixels_outer],
#            header='inner_mean\t background_mean\t backgroudn_std\t inner_no_poitns\t outer_no_points')






