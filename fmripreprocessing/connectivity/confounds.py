import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level.design_matrix import _convolve_regressors
from nilearn.interfaces.fmriprep import load_confounds_strategy
import nilearn.image as nim

def load_confounds(
    events,
    fmri_path,
    strategy="simple",
    task=True,
    reg_type="FIR",
    gs="basic",
    TR=1,
    task_id=None,
):
    if strategy is not None:
        nuisance_confounds, _ = load_confounds_strategy(
        fmri_path, strategy, global_signal=gs
        )
    else:
        nuisance_confounds = pd.concat([pd.DataFrame(nim.high_variance_confounds(fmri_path)), pd.read_table("/homes/a19lamou/fmri_data_proc/data/language_loc/fMRI-language-localizer-demo-dataset/derivatives/sub-01/func/sub-01_task-languagelocalizer_desc-confounds_regressors.tsv")], axis=1)
    if isinstance(fmri_path, str):
        func_img = nib.load(fmri_path)
    else:
        func_img = fmri_path
    # corection since we do slice timing correction see
    # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
    end_time = (func_img.shape[-1] - 1 + TR) * TR
    frame_times = np.linspace(0, end_time, func_img.shape[-1]) + TR / 2

    task_regressor = get_task_function(events, frame_times, task_id=task_id)

    if task and reg_type == "FIR":
        fir_cond_mat = convert_task_timing_to_FIR(
            events, frame_times, firLag=2
        )
        for i in range(fir_cond_mat.shape[1]):
            nuisance_confounds[f"Task{i}"] = fir_cond_mat[:, i]
    elif task and reg_type != "FIR":
        nuisance_confounds["Task"] = task_regressor
    return nuisance_confounds, task_regressor


def get_task_function(events, frame_times, task_id):
    if isinstance(events, str):
        events = pd.read_table(
            events, usecols=["onset", "duration", "trial_type"]
        )
    matrix, names = _convolve_regressors(
        events=events, hrf_model="glover", frame_times=frame_times
    )
    task_regressor = matrix[:, names.index(task_id)]
    return task_regressor


def convert_task_timing_to_FIR(events, frame_times, firLag=2):
    if isinstance(events, str):
        events = pd.read_table(
            events, usecols=["onset", "duration", "trial_type"]
        )
    n_times_points = len(frame_times)

    block_onsets = np.array(
        events[events["trial_type"] != "Rest"]["onset"], dtype=int
    )

    block_offsets = block_onsets + np.array(
        events[events["trial_type"] != "Rest"]["duration"], dtype=int
    )

    block_length = np.max(block_offsets - block_onsets) + firLag

    n_blocks = len(block_onsets)

    fir_cond_mat = np.zeros((n_times_points, block_length))

    for block in range(n_blocks):
        trcount = block_onsets[block]
        for i in range(block_length):
            if trcount <= n_times_points:
                fir_cond_mat[trcount, i] = 1
                trcount = trcount + 1
            else:
                continue
    return fir_cond_mat
