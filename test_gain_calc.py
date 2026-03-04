# test gain make_gain_primitive.py
# activate uv env
# source /Users/ashbake/Documents/Research/Projects/HISPEC/HISPEC_DRP/hispec/bin/activate
import glob
import sys
import numpy as np
from astropy.io import fits
import matplotlib.pylab as plt
from astropy.time import Time


DATA_FOLDER = '/Users/ashbake/Documents/Research/Projects/HISPEC/_DATA/h2rg_images/20260226/'
DRP_PATH = '/Users/ashbake/Documents/Research/Projects/HISPEC/HISPEC_DRP/hispecdrp/'

sys.path.append(DRP_PATH)

from hispecdrp.primitives import make_gain_primitive as mgp
from hispecdrp import datamodels


# prepare files into DRP format. 
# eg from simulations/create_ramp.py:

# data_np: 
def make_DRP_compatible_dict(filename):
    """
    load file at filename and open it conform UTR files 
    from camerad into _outputs dictionary to feed DRP

    subject to change when camerad *_cube.fits file format changes - ashley
    """
    f = fits.open(filename)
    data, header = f[0].data, f[0].header
    
    # make data
    data_np = np.float32(data[1:, :, :] - data[0])# subtract reset frame and discard it
    n_reads = data_np.shape[0]
    nx, ny = data_np.shape[1], data_np.shape[2]
    dq_raw_np = np.zeros(shape=(n_reads, nx, ny), dtype=np.uint8)
                         
    # define meta data
    n_channels = 4
    T_PIX = 9e-6 # us # ACF should be labeled with this, but currenlty (2/27/26) it's guess work
    clock_rate = 1/T_PIX # Hz
    mjd_start = header['DATE']
    t = Time(header['DATE'], format='isot', scale='utc')
    mjd_start = t.mjd
    #clock_rate = 1/(readtime/((nx*ny)//n_channels)) # in Hz
    _meta = dict(clock_rate=clock_rate,
             mjd_start=mjd_start,
             n_channels=n_channels,
             channels_are_vertical=True,
             n_refpixs = 2)
    
    #_outputs = dict(data=data_np, dq_raw=dq_raw_np, meta=_meta)

    # make into ramp model to feed gain code:
    ramp_model = datamodels.RampModel(data=data_np, dq_raw=dq_raw_np,
                                        meta=_meta)
    ramp_model.meta.filename=filename

    return ramp_model


if __name__=='__main__':
    files = glob.glob(DATA_FOLDER + '*cube.fits')

    # process all data
    ramp_model_seq = []
    for i, filename in enumerate(files):
        ramp_model = make_DRP_compatible_dict(filename)
        ramp_model_seq.append(ramp_model)
        print(f'loaded {i} of {len(files)}:', filename)

    # plot ramps for one pixel for all ramps
    plt.figure()
    for ramp_model in ramp_model_seq:
        plt.plot(ramp_model.data[:,1000,1000],'.')

    # get ready for gain calc
    mg = mgp.MakeGainPrimitive()

    # set starting params
    mg.init_gain = 2.0 # e-/DN
    mg.init_RN   = 5.0 # DN 
    mg.mode      = 'per_pixel' # 'per_pixel' | 'channel' | 'full_detector'
    mg.method    = 'grid'
    mg.init_delta_gains = np.linspace(-1.9, 2, 20)
    mg.init_delta_RNs   = np.linspace(-1,1,10)
    mg.fix_RN           = False
    mg.countratesidentical = False
    mg.max_cores = 1

    # should be using apply!
    output_gain = mg.apply(input=ramp_model_seq,
                           save_result=True,
                           init_RN=7.0)

    # save data model
    output_gain['output'].save(output_dir=DATA_FOLDER,filename='test_gain_fit_countratesNOTsame.fits')

    plt.figure()
    plt.imshow(output_gain['output'].GAIN)
    plt.colorbar()
    plt.title('Derivced Gains')
    plt.savefig(DATA_FOLDER + 'test_gain_fit_countratesNOTsame.png')

    plt.figure()
    plt.imshow(output_gain['output'].COV[0,0]) # plot errors
    plt.title('Gain Errors COV[0,0]')
    plt.colorbar()

    # test pipeline here really quickly
    from hispecdrp.pipelines import process_darks_pipeline
    test = process_darks_pipeline.ProcessDarksPipeline()
    test.input = ramp_model_seq
    pipe_out = test._perform()


    