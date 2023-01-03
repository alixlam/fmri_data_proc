import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix

def run_first_level(event_file,  fmri_session, brain_mask,task_name = "TaskNF", TR = 1):
    """Run first level GLM n 

    Parameters
    ----------
    event_file : str
        path to the event file (.tsv file)
    fmri_session :
        _description_
    brain_mask : _type_
        _description_
    task_name : str, optional
        _description_, by default "TaskNF"
    TR : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    events = pd.read_table(event_file)
    

    func_img = nib.load(fmri_session)
    # Build design matrix
    frame_times = np.arange(func_img.shape[-1]* TR)
    columns_name = ['onset', 'duration', 'trial_type']
    events = events.drop(columns=[c for c in events if c not in columns_name])
    design_matrix = make_first_level_design_matrix(
        frame_times, events, drift_model = 'polynomial', drift_order =3, hrf_model='glover'
    )

    # Instantiate the first level model
    glm = FirstLevelModel(t_r = TR, mask_img=brain_mask, smoothing_fwhm=4)

    # Fit the GLM
    glm.fit(fmri_session, design_matrices = [design_matrix])

    #Compute contrasts
    z_maps = {}
    for contrast in ["Task", "Task - Rest"]:
        z_map = glm.compute_contrast(contrast)
        z_maps[contrast] = z_map
    return z_maps

