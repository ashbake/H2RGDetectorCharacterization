import os, sys
import glob
import numpy as np
import matplotlib.pylab as plt

from data_utils import make_DRP_compatible_dict
from dotenv import load_dotenv
load_dotenv()  # Load credentials for calibration DB at Keck
import logging

DATA_FOLDER = '/Users/ashbake/Documents/Research/Projects/HISPEC/_DATA/h2rg_images/20260226/'
DRP_PATH = '/Users/ashbake/Documents/Research/Projects/HISPEC/HISPEC_DRP/hispecdrp/'

sys.path.append(DRP_PATH)

from hispecdrp import datamodels
from hispecdrp.primitives import (DQInitPrimitive, MakeBiasPrimitive, CheckSaturationPrimitive, NonlinearCorrectionPrimitive,
                           JumpDetectionPrimitive, FitRampPrimitive, DarkSubtractionPrimitive,
                           BiasSubtractionPrimitive, DetectorFlatCorrectionPrimitive, GainCorrectionPrimitive,
                           MakeReadnoisePrimitive, MakeDarkPrimitive, RegisterCalibrationPrimitive,
                           CoaddFramesPrimitive)
from typing import Sequence

# Configure calibration cache directory if not already set in shell environment
os.environ['KOA_CALIBRATION_CACHE'] = '/Users/ashbake/Documents/Research/Projects/HISPEC/_DATA/DRP_data/'


### MAKE RAMP INPUT DIC
# I think I need DATAMODL Keywork in fits for the pipeline to load them
files = glob.glob(DATA_FOLDER + '*cube.fits')[0:10]

ramp_model_seq = []
for i, filename in enumerate(files):
    ramp_model = make_DRP_compatible_dict(filename)
    ramp_model_seq.append(ramp_model)
    print(f'loaded {i} of {len(files)}:', filename)


# RUN [my own] PIPELINE
#DQInitPrimitive
bias_prim = MakeBiasPrimitive(input = ramp_model_seq)
bias = bias_prim.apply(input=ramp_model_seq,max_cores=1)

# this seems to not skip everything, need to write my own pipeline??