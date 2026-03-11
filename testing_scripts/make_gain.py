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

from hispecdrp.primitives import make_gain_primitive as mgp


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

    # plot ramps for one pixel for all ramps
    plt.figure()
    for ramp_model in ramp_model_seq:
        plt.plot(ramp_model.data[:,10,10],'.')

    # get ready for gain calc
    mg = mgp.MakeGainPrimitive()

    # set starting params
    mg.init_gain = 2.0 # e-/DN
    mg.init_RN   = 5.0 # DN 
    mg.mode      = 'per_pixel' # 'per_pixel' | 'channel' | 'full_detector'
    mg.method    = 'grid'
    mg.init_delta_gains = np.linspace(-1.9, 2, 20)
    mg.init_delta_RNs   = np.linspace(-1,1,10)
    mg.fix_RN           = True
    mg.countratesidentical = True
    mg.max_cores = 1

    # should be using apply!
    output_gain = mg.apply(input=ramp_model_seq,
                           save_result=True,
                           output_dir=CAL_FOLDER,
                           init_RN=7.0)

    # save data model
    #output_gain['output'].save(output_dir=DATA_FOLDER)#,filename='test_gain_fit_countratesNOTsame.fits')

    plt.figure()
    plt.imshow(output_gain.gain)
    plt.colorbar()
    plt.title('Derived Gains')
    plt.savefig(DATA_FOLDER + 'test_gain_fit_countratesNOTsame.png')

    plt.figure()
    plt.imshow(output_gain.COV[0,0]) # plot errors
    plt.title('Gain Errors COV[0,0]')
    plt.colorbar()



    