import nibabel as nib
import nilearn.image as nim
import numpy as np
import pandas as pd
from nilearn.glm.first_level import (
    FirstLevelModel,
    make_first_level_design_matrix,
)
from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy

from fmripreprocessing.utils.data import *


def run_first_level(
    event_file,
    fmri_session,
    brain_mask,
    task_name="TaskNF",
    TR=1,
    output_type="z_score",
    motion=True,
    path_to_confounds=None,
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
        path_to_confounds = fmri_session
    else:
        if motion:
            assert (
                path_to_confounds is not None
            ), "You need to provide the path to the confounds"
        func_img = fmri_session
    # Build design matrix
    # corection since we do slice timing correction see
    # https://reproducibility.stanford.edu/slice-timing-correction-in-fmriprep-and-linear-modeling/
    frame_times = np.arange(func_img[0].shape[-1] * TR) + TR / 2
    columns_name = ["onset", "duration", "trial_type"]
    events = events.drop(columns=[c for c in events if c not in columns_name])
    if motion:
        motion_confonds, _ = load_confounds(
            path_to_confounds, strategy=["motion"], motion="basic"
        )
        if not isinstance(motion_confonds, list):
            motion_confonds = [motion_confonds]
        reg = [np.array(confonds) for confonds in motion_confonds]
        reg_names = list(motion_confonds[0].columns)
    else:
        reg = [None] * len(func_img)
        reg_names = None
    design_matrices = []
    for conf in reg:
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events,
            drift_model="polynomial",
            drift_order=3,
            hrf_model="glover",
            add_regs=conf,
            add_reg_names=reg_names,
        )
        design_matrices.append(design_matrix)

    # Instantiate the first level model
    glm = FirstLevelModel(t_r=TR, mask_img=brain_mask, smoothing_fwhm=6)

    # Fit the GLM
    glm.fit(func_img, design_matrices=design_matrices)

    # Compute contrasts
    z_maps = {}
    for contrast in [f"{task_name}", f"{task_name} - Rest", "Rest"]:
        z_map = glm.compute_contrast(contrast, output_type=output_type)
        z_maps[contrast] = z_map
    return z_maps, glm


def run_first_model_dataset(
    path_to_dataset="/users2/local/alix/out2",
    path_to_events="/users2/local/alix/XP2",
    subjects_to_include=["sub-xp201"],
    task_name="NF",
    regression="simple",
    space= "MNI152NLin2009cAsym_res-2",
    run_ids=[1, 2, 3],
):
    contrast_maps = {}
    for sub in subjects_to_include:
        functional_paths = get_subjects_functional_data(
            path_to_dataset, sub, task=task_name, run_ids=run_ids, space=space,
        )
        print(functional_paths)
        if "MI" in task_name:
            events = get_event_file(
                path_to_data=path_to_events, task_name=task_name + run_ids
            )
        elif "NF" in task_name:
            events = (
                get_event_file(path_to_data=path_to_events, task_name="1dNF")
                if "1dNF" in functional_paths[0]
                else get_event_file(
                    path_to_data=path_to_events, task_name="2dNF"
                )
            )
        brain_mask = get_subject_brain_mask_from_T1(path_to_dataset, sub, space = space)

        if regression is not None:
            clean_img = []
            confounds, _ = load_confounds_strategy(
                functional_paths, regression, global_signal="basic"
            )
            if not isinstance(confounds, list):
                confounds = [confounds]
            for img, conf in zip(functional_paths, confounds):
                clean_img.append(
                    nim.clean_img(
                        img, confounds=conf, detrend=False, standardize=False
                    )
                )
        else:
            clean_img = functional_paths

        task_name_glm = "TaskNF" if "NF" in task_name else "TaskMI"
        contrast, glm = run_first_level(
            event_file=events,
            fmri_session=clean_img,
            brain_mask=brain_mask,
            task_name=task_name_glm,
            output_type="stat",
            motion=True,
            path_to_confounds=functional_paths,
        )

        contrast_maps.update({f"{sub}": contrast})
    return contrast_maps
