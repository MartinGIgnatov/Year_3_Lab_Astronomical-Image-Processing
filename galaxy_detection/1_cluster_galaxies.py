import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from astropy.io import fits
from kernels import convolve,EDGEDETECTION,GAUSSIANBLUR,gaussian
import astropy.visualization as visualization
import time
from sys import stdout
plt.style.use('mystyle-2.mplstyle')

def show(img):
    plt.figure()
    imshow(img)
    plt.show()

zscale=visualization.ZScaleInterval()

# import original image
filename = "A1_mosaic_nostar.fits" # with frame but no star
hdulist=fits.open(filename)
image = hdulist[0].data
hdulist.close()

# import mask
filename = "mask.fits" # with frame but no star
hdulist=fits.open(filename)
convolution_threshold = hdulist[0].data
hdulist.close()

# remove large star at the center and borders from mask
ignore_area=np.ones(convolution_threshold.shape)
ignore_area[2900:3500,1250:1642]=0
ignoreborder=100
# ignore_area[:ignoreborder,:]=0 # ignore top border
# ignore_area[-ignoreborder:,:]=0 # ignore bottom border
# ignore_area[:,:ignoreborder]=0 # ignore left border
# ignore_area[:,-ignoreborder:]=0 # ignore right border
convolution_threshold=convolution_threshold*ignore_area


# select patch from whole image (used for testing)
patch=image[1000:1200,1000:1200]
patchthreshold=convolution_threshold[1000:1200,1000:1200]

show(zscale(image))
show(patchthreshold)


#%%
# =============================================================================
# start algorithm, this can take about 1h !
# =============================================================================

# uncomment to use another patch useful for testing
# patch=np.reshape(np.arange(64),(8,8))
# patchmask=np.zeros(patch.shape)
# patchmask[1:3,1:3]=1
# patchmask[5:-1,5:-1]=1

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
        
        stdout.write(f"\rGalaxy number: {i} / {len(np.argwhere(tempmask==1))}      ")
        stdout.flush()
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


timestart=time.time()
galaxylist=np.array(index_galaxies(image, convolution_threshold,3))
# galaxylist=np.array(index_galaxies(patch, patchmask,1)) # uncommmento to test algorithm on patch
timeend=time.time()
timetotal=timeend-timestart
print(f'\n\nTime taken: {timetotal}')
print(f'first galaxies: {galaxylist}')




# uncomment to save the result (be careful not to overwrite existing data!)
# np.savetxt('located_galaxies_00/galaxypositions-final.txt',galaxylist)


# show(patchmask)
# show(zscale(patch))
# print(f'{galaxylist}')



