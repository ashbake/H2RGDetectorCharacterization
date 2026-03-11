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

DATA_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/20260226/utr_10reads/'

DRP_PATH = '/Users/ashleybaker/Documents/HISPEC/HISPEC_DRP/hispecdrp/'
CAL_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/calibrations/'

sys.path.append(DRP_PATH)

from hispecdrp.primitives import make_rn_primitive as mrnp


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

    # get ready for gain calc
    rn = mrnp.MakeReadnoisePrimitive()

    rn.gain =  CAL_FOLDER + 'HB.20260226.03605.43.mastercal-gain.fits'

    rn.gain =  CAL_FOLDER + 'HB.20260226.03605.43.mastercal-gain.fits'

    # should be using apply!
    output_rn = rn.apply(input=ramp_model_seq,
                           save_result=True,
                           output_dir=CAL_FOLDER,
                           init_RN=7.0,
                           mode='per_pixel',
                           method='grid',
                           init_delta_RNs=np.linspace(-6.9,10,10))

    output_rn.rn
