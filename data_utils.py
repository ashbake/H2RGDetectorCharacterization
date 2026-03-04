import sys
import numpy as np
from astropy.io import fits
from astropy.time import Time

DRP_PATH = '/Users/ashbake/Documents/Research/Projects/HISPEC/HISPEC_DRP/hispecdrp/'
sys.path.append(DRP_PATH)

from hispecdrp import datamodels


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
    dq_np = np.zeros(shape=(n_reads, nx, ny), dtype=np.uint32)

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
    ramp_model = datamodels.RampModel(data=data_np, dq_raw=dq_raw_np, dq=dq_np,
                                        meta=_meta)
    ramp_model.meta.filename=filename

    return ramp_model
