# this just runs the MakeBiasPrimitive
# fails at dq map for some reason, to be debugged
# now i have commented that part out in makebiasprimitive to skip dq generation
# https://github.com/oirlab/HISPEC_DRP/issues/173
import os, sys
import glob
import numpy as np
import matplotlib.pylab as plt

from data_utils import make_DRP_compatible_dict
from dotenv import load_dotenv
load_dotenv()  # Load credentials for calibration DB at Keck

DATA_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/20260226/utr_10reads/'
DRP_PATH = '/Users/ashleybaker/Documents/HISPEC/HISPEC_DRP/hispecdrp/'
CAL_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/calibrations/'

sys.path.append(DRP_PATH)

from hispecdrp import datamodels
from hispecdrp.primitives import (DQInitPrimitive, MakeBiasPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
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
    ramp_model = make_DRP_compatible_dict(filename,subtract_reset=False)
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

# BIAS
bias_prim = MakeBiasPrimitive(bias_threshold=5000,
                            output_dir=CAL_FOLDER)
bias = bias_prim.apply(input=ramp_model_seq,max_cores=1)

plt.imshow(bias.bias)
bias_prim.save_output()

# bias_sub



