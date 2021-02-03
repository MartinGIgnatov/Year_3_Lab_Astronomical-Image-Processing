"""
Contains the function that generates random samples. Used to check the
ipothesis of the bias from the border
"""
import numpy as np
import numpy.random
import scipy as sp
import scipy.interpolate
import sys 
import matplotlib.pyplot as plt
plt.style.use('mystyle-2.mplstyle')

def get_points(N, grad, intercept, lower_boundary, upper_boundary):

    def line(m):
        return m * grad + intercept
    
    normalisation = 1/(10**(line(upper_boundary))- 10**(line(lower_boundary)))

    uniform_random  = np.random.uniform(size = N)
    random_magnitude = (np.log10(uniform_random/normalisation + 10**(line(lower_boundary))) \
                        - intercept)/grad
    return random_magnitude


low = 7
high = 20
a = get_points(100000,0.36111444, -2.67273998,low,high)

binrange = np.arange(low,high,0.3)

totalN = []

for bn in binrange:
    totalN.append(len(np.argwhere(a <= bn)))
totalN = np.array(totalN)
print(f'total N: {totalN}')

skipfirst = 5
binrange = binrange[5:]
totalN = totalN[5:]

plt.plot(binrange,np.log10(totalN))


    
    
    
