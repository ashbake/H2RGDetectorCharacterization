import os, sys
import glob
import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits

from data_utils import make_DRP_compatible_dict
DATA_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/20260303/darks_warmup/'


### MAKE RAMP INPUT DIC

if __name__=='__main__':
    # Load data
    files = np.sort(glob.glob(DATA_FOLDER + '*cube.fits'))
    # process all data
    ramp_model_seq = []
    detector_temps = []
    for i, filename in enumerate(files):
        ramp_model = make_DRP_compatible_dict(filename,subtract_reset=True,subframe=[700,1300])
        ramp_model_seq.append(ramp_model)
        print(f'loaded {i} of {len(files)}:', filename)
        hdr = fits.getheader(filename)
        detector_temps.append(hdr['DETTEMP'])


    # plot
    avg_value = []
    for ramp_model in ramp_model_seq:
        avg_value.append(np.median(ramp_model.data[10]))
    
    plt.figure()
    plt.plot(detector_temps,avg_value)