import os, sys
import glob
import numpy as np
import matplotlib.pylab as plt

from data_utils import make_DRP_compatible_dict
from dotenv import load_dotenv
load_dotenv()  # Load credentials for calibration DB at Keck
import logging

DATA_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/20260226/utr_10reads/'
#DATA_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/20260303/saturated_flats_UTR/'
CAL_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/calibrations/'

DRP_PATH = '/Users/ashleybaker/Documents/HISPEC/HISPEC_DRP/hispecdrp/'

sys.path.append(DRP_PATH)

from hispecdrp import datamodels
from hispecdrp.primitives import (DQInitPrimitive, MakeBiasPrimitive, CheckSaturationPrimitive,MakeNonLinPrimitive,NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive, FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive, DetectorFlatCorrectionPrimitive, GainCorrectionPrimitive,
                           MakeReadnoisePrimitive, MakeDarkPrimitive, RegisterCalibrationPrimitive,
                           CoaddFramesPrimitive)
from typing import Sequence

# Configure calibration cache directory if not already set in shell environment
os.environ['KOA_CALIBRATION_CACHE'] = '/Users/ashleybaker/Documents/HISPEC/_data/drp_test_data/'


### MAKE RAMP INPUT DIC
# I think I need DATAMODL Keywork in fits for the pipeline to load them
files = glob.glob(DATA_FOLDER + '*cube.fits')[0:10]

ramp_model_seq = []
for i, filename in enumerate(files):
    ramp_model = make_DRP_compatible_dict(filename,subtract_reset=True)
    ramp_model_seq.append(ramp_model)
    print(f'loaded {i} of {len(files)}:', filename)


# RUN [my own] PIPELINE

"""     'dq_init': DQInitPrimitive,
        'make_bias': MakeBiasPrimitive,
        'register_bias_cal' : RegisterCalibrationPrimitive,
        'saturation_check': CheckSaturationPrimitive,
        'bias_sub': BiasSubtractionPrimitive,
        'nonlin_corr': NonlinearCorrectionPrimitive,
        'jump_detec': JumpDetectionPrimitive,
        'make_rn': MakeReadnoisePrimitive,
        'register_rn_cal' : RegisterCalibrationPrimitive,
        'ramp_fit': FitRampPrimitive,
        'coadd': CoaddFramesPrimitive,
        'make_dark': MakeDarkPrimitive,
        'register_dark_cal' : RegisterCalibrationPrimitive,
        'dark_sub': DarkSubtractionPrimitive,
        'gain_corr': GainCorrectionPrimitive,
        'detflat_corr': DetectorFlatCorrectionPrimitive,"""

# nonlin_corr
nlin = MakeNonLinPrimitive()
nlin.gain = CAL_FOLDER + 'HB.20260226.03605.43.mastercal-gain.fits'
nlin.bias = CAL_FOLDER + 'HB.20260226.03605.43.mastercal-bias.fits'
nlin.rn = CAL_FOLDER + 'HB.20260226.03605.43.mastercal-rn.fits'
nlin.saturation = CAL_FOLDER + 'HB.20260303.24694.88.mastercal-saturation.fits'

nlin.apply(input = ramp_model_seq,
           output_dir=CAL_FOLDER,
           mode='per_pixel') 
# can run in mode full_detector with a sub region to see if it works TODO
nlin.save_output()



