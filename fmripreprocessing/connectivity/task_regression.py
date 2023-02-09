from nilearn.glm.first_level import FirstLevelModel
import pandas as pd
import nibabel as nib
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.

def get_residuals(
    event_file,
    fmri_session,
    brain_mask,
    task_name="TaskNF",
    TR=1,
):
    """Run first level GLM

    Parameters
    ----------
    event_file : str
        path to the event file (.tsv file)
    fmri_session : str
        path to functional mri
    brain_mask : str
        path to brain mask
    task_name : str, optional
        name of the task, by default "TaskNF"
    TR : int, optional
        TR , by default 1

    Returns
    -------
    dict
        dict containing the constrasts
    """

    events = pd.read_table(event_file)
    if not isinstance(fmri_session, list):
        fmri_session = [fmri_session]

    if isinstance(fmri_session[0], str):
        func_img = [nib.load(sess) for sess in fmri_session]
    else:
        func_img = fmri_session
    # Build design matrix
    # corection since we do slice timing correction see
    # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
    frame_times = np.arange(func_img[0].shape[-1] * TR) + TR/2
    columns_name = ["onset", "duration", "trial_type"]
    events = events.drop(columns=[c for c in events if c not in columns_name])
    design_matrices = []
    for img in func_img:
		design_matrix = make_first_level_design_matrix(
			frame_times,
			events,
			drift_model=None,
			hrf_model="spm",
		)
		design_matrices.append(design_matrix)

    # Instantiate the first level model
    glm = FirstLevelModel(t_r=TR, mask_img=brain_mask, smoothing_fwhm=6,signal_scaling = False,minimize_memory=False)

    # Fit the GLM
    glm.fit(func_img, design_matrices=design_matrices)

    return glm.residuals()

def get_fc_roi(atlas, brain_mask, **conn_args):
    