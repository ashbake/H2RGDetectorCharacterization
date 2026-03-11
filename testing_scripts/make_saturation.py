# test gain make_gain_primitive.py
# activate uv env
# source /Users/ashbake/Documents/Research/Projects/HISPEC/HISPEC_DRP/hispec/bin/activate
import glob
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
from astropy.time import Time

from data_utils import make_DRP_compatible_dict

DATA_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/20260303/saturated_flats_UTR/'
DRP_PATH = '/Users/ashleybaker/Documents/HISPEC/HISPEC_DRP/hispecdrp/'
CAL_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/calibrations/'

sys.path.append(DRP_PATH)

from hispecdrp.primitives import make_saturation_primitive as msp


# prepare files into DRP format. 
# eg from simulations/create_ramp.py:



if __name__=='__main__':
    files = glob.glob(DATA_FOLDER + '*cube.fits')

    # process all data
    ramp_model_seq = []
    for i, filename in enumerate(files):
        ramp_model = make_DRP_compatible_dict(filename,subtract_reset=True)
        ramp_model_seq.append(ramp_model)
        print(f'loaded {i} of {len(files)}:', filename)

    # do saturation calculation
    ms = msp.MakeSaturationPrimitive()
    output_sat = ms.apply(input = ramp_model_seq,
                          offset = 100,
                          max_cores=1,
                          save_result=True,
                          output_dir=CAL_FOLDER)

    plt.figure()
    plt.imshow(output_sat.data_dict['sat_thresh'])
    plt.colorbar()
    plt.title('Saturation Map')





    