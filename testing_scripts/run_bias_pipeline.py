# this runs the make bias code from the pipeline level. similar to test_make_bias.py
from hispecdrp import ProcessBiasPipeline
from hispecdrp import datamodels
import sys
import glob
from data_utils import make_DRP_compatible_dict

DATA_FOLDER = '/Users/ashleybaker/Documents/HISPEC/_data/h2rg_images/20260226/utr_10reads/'
DRP_PATH = '/Users/ashleybaker/Documents/HISPEC/HISPEC_DRP/hispecdrp/'

sys.path.append(DRP_PATH)

files = glob.glob(DATA_FOLDER + '*cube.fits')[0:10]

ramp_model_seq = []
for i, filename in enumerate(files):
    ramp_model = make_DRP_compatible_dict(filename,subtract_reset=False)
    ramp_model_seq.append(ramp_model)
    print(f'loaded {i} of {len(files)}:', filename)

pipe_max_cores=1
prim_args = {
    'dq_init': {
        'run': True,
        'max_cores': pipe_max_cores,
    },
    'make_bias': {
        'run': True,
        'max_cores': pipe_max_cores,
        'bias_threshold': 6000,
        'kTC_threshold': 1000,
    },
    'register_bias_cal': {
        'run': True,
    }
}


dqinit_map = ramp_model_seq[0].dq
# set row to bad
# dqinit_map[5, :] |= DQ_FLAGS["DO_NOT_USE"]
dqinit_model = datamodels.DQModel(dq=dqinit_map)

# Setup and call the stage 0 pipeline
bias_pipe = ProcessBiasPipeline()

# Set any primitive args

bias_pipe.dq_init.dq = dqinit_model
print(ramp_model_seq[0].data[0])
# Run pipeline
model_result = bias_pipe.apply(ramp_model_seq, prim_args=prim_args)
