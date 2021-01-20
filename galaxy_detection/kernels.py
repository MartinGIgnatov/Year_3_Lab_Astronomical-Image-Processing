import numpy as np
import scipy.signal
import scipy as sp



# Try convolution

EDGEDETECTION = np.array([[-1,-1,-1],
                  [-1,8,-1],
                  [-1,-1,-1]])

GAUSSIANBLUR= np.array([[1,2,1],
                       [2,4,2],
                       [1,2,1]]) / 16


SOBELX= np.array([[-1,0,1],
                  [-2,0,2],
                  [-1,0,1]])

SOBELY= np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])

def convolve(img,kernel):
    convimage = sp.signal.convolve2d(img,kernel,mode='same',boundary='symm')
    return convimage
    
def gaussian(sigma=0.85, N=3):
    x = np.arange(N)-N//2
    y = np.arange(N)-N//2
    xx,yy=np.meshgrid(x,y)
    gaussiankern=np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    gaussiankern=gaussiankern/np.sum(gaussiankern)
    return gaussiankern



# #%%
# print(GAUSSIANBLUR)
# print(gaussian(0.85))