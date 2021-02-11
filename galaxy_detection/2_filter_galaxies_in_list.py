"""Filter the list of galaxies for brightness and location """

import numpy as np
import matplotlib.pyplot as plt
import galaxy_list_filter as galaxyfilter
plt.style.use('mystyle-2.mplstyle')


# Filter Galaxy list

radius_inner = 14 # 15
radius_outer = 24 # 30

ignore_border=150 # avoid galaxies too close to border

# load raw galaxylist
galaxylist_raw = np.loadtxt('located_galaxies_00/galaxypositions-final.txt')

# perform filtering algorithm
galaxylist = galaxyfilter.clean_list_galaxies(galaxylist_raw,min_brightness=3465,
                                              max_brightness=35000,ignore_border=ignore_border,radius=radius_inner)

# save results
# np.savetxt('galaxy_brightness_analysis_results/galaxylist_cleaned.txt',galaxylist,header='row\t col\t maxpix\t no. pix')
