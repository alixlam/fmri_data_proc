from nilearn import image as nim
from nilearn.interfaces.fmriprep import load_confounds_strategy
from nilearn.glm.first_level.design_matrix import _convolve_regressors
import pandas as pd
import numpy as np
import os, glob
import warnings

def confound_regression(func_img, confound_strategy = None, gs=None, confounds= None, mask_img=None, **cleanargs):
    
    # Load confounds
    if confounds is not None:
        pass
    else:
        if confound_strategy is not None:
            confounds, _ = load_confounds_strategy(func_img, denoise_strategy=confound_strategy, global_signal=gs)
        else:
            confounds_compcor = pd.DataFrame(nim.high_variance_confounds(func_img))
            confounds_mouv_file = glob.glob(os.path.join(os.path.dirname(func_img), "*regressors*"))
            if len(confounds_mouv_file) > 0:
                confounds = pd.concat([confounds_compcor, pd.read_table(confounds_mouv_file[0])], axis=1)
            else:
                confounds = confounds_compcor
    #clean_img
    denoised_fmri = nim.clean_img(
        func_img, confounds=confounds, mask_img = mask_img, **cleanargs
    )
    
    return denoised_fmri


def get_task_function(events, TR=None, task_id="NF", ntime=300):
    if isinstance(events, str):
        events = pd.read_table(
            events, usecols=["onset", "duration", "trial_type"]
        )
    if TR is None:
        warnings.warn("No TR was provided, assuming TR = 1s")
        
    end_time = (ntime - 1 + TR) * TR
    frame_times = np.linspace(0, end_time, ntime) + TR / 2
    matrix, names = _convolve_regressors(events=events, frame_times=frame_times, hrf_model="glover")
    
    task_regressor = matrix[:, names.index(task_id)]
    return task_regressor

def convert_task_timing_to_FIR(events, ntime, task_id=None, firLag=10):
    if isinstance(events, str):
        events = pd.read_table(
            events, usecols=["onset", "duration", "trial_type"]
        )
    

    block_onsets = np.array(
        events[events["trial_type"] == task_id]["onset"], dtype=int
    )

    block_offsets = block_onsets + np.array(
        events[events["trial_type"] == task_id]["duration"], dtype=int
    )

    block_length = np.max(block_offsets - block_onsets) + firLag

    n_blocks = len(block_onsets)

    fir_cond_mat = np.zeros((ntime, block_length))

    for block in range(n_blocks):
        trcount = block_onsets[block]
        for i in range(block_length):
            if trcount <= ntime:
                fir_cond_mat[trcount, i] = 1
                trcount = trcount + 1
            else:
                continue
    return fir_cond_mat

def task_regression(func_img, events, regression_strategy= "FIR", firlag=None, TR=1,  task_id=None, mask_img=None):
    if regression_strategy == "FIR":
        task_regressor = convert_task_timing_to_FIR(events, func_img.shape[-1], task_id, firLag=firlag)
    else:
        task_regressor= get_task_function(events, task_id=task_id, ntime=func_img.shape[-1], TR=TR)
    cleaned_img = confound_regression(func_img, confounds= task_regressor,  mask_img=mask_img,detrend=False, standardize=False,)
    
    return cleaned_img