import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time
import glob

DRP_PATH = '/Users/ashbake/Documents/Research/Projects/HISPEC/HISPEC_DRP/hispecdrp/'
sys.path.append(DRP_PATH)

from hispecdrp import datamodels


def make_DRP_compatible_dict(filename,subtract_reset=False,subframe=[0,2048]):
    """
    load file at filename and open it conform UTR files 
    from camerad into _outputs dictionary to feed DRP

    subject to change when camerad *_cube.fits file format changes - ashley

    inputs
    ------
    filename [str]
        name of camerad generated fits image to open
    subtract_reset [bool]
        whether to subtract the reset frame in that image from all reads
    subframe [tuple]
        min and max of subframe to apply. currently applies to both axes (square frame only). default [0,2048]

    outputs:
    --------
    HISPEC DRP Ramp Model
    """
    f = fits.open(filename)
    data, header = f[0].data, f[0].header
    
    # make data
    if subtract_reset:
        data_np = np.float32(data[:, :, :] - data[0])# subtract reset frame
    else:
        data_np = np.float32(data)

    n_reads = data_np.shape[0]
    nx, ny = data_np.shape[1], data_np.shape[2]
    
    # make bad pixel maps
    dq_raw_np = np.zeros(shape=(n_reads, nx, ny), dtype=np.uint8) # unique to pixels and reads
    dq_np = np.zeros(shape=(n_reads, nx, ny), dtype=np.uint32)  # unique to pixels, applies to all reads

    # first row is bad
    dq_np[:,0, :] = 1

    # first frame is bad bc if bias corrercted - it is offset from the rest
    if subtract_reset: dq_np[0,:, :] = 1
    
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
    
    # make into ramp model
    mn, mx = subframe
    ramp_model = datamodels.RampModel(data=data_np[:,mn:mx,mn:mx], dq_raw=dq_raw_np[:,mn:mx,mn:mx], dq=dq_np[:,mn:mx,mn:mx],
                                        meta=_meta)
    ramp_model.meta.filename=filename

    return ramp_model


# def load files into dictionary
def get_ramp_seq(DATA_FOLDER,subtract_reset=True,max_files=-1,subframe=[0,2048]):
    files = np.sort(glob.glob(DATA_FOLDER + '*cube.fits')[0:max_files])

    # process all data
    ramp_model_seq = []
    for i, filename in enumerate(files):
        ramp_model = make_DRP_compatible_dict(filename,subtract_reset=subtract_reset,subframe=subframe)
        ramp_model_seq.append(ramp_model)
        print(f'loaded {i} of {len(files)}:', filename)

    return ramp_model_seq