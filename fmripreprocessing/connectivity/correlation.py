import nibabel as nib
import nilearn.image as nim
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from tqdm import tqdm
from nilearn.interfaces.bids import get_bids_files

from fmripreprocessing.connectivity.confounds import load_confounds
from fmripreprocessing.utils.data import *
from fmripreprocessing.utils.masks import intersect_multilabel


def get_fc_roi_single(
    atlas,
    brain_mask,
    fmri,
    prob=True,
    confounds=None,
    task_regressor=None,
    only_task=True,
    cropped_mask=True,
    separate_runs=True,
    run_ids=[1, 2, 3],
    **conn_args,
):
    connectivity_measure = ConnectivityMeasure(**conn_args)
    if cropped_mask:
        mask_cropped = atlas
    else:
        mask_cropped = intersect_multilabel(brain_mask, atlas)
    if not prob:
        masker = NiftiLabelsMasker(
            mask_cropped, mask_img=brain_mask, standardize=True, verbose=0
        )

    else:
        masker = NiftiMapsMasker(
            atlas.maps, mask_img=brain_mask, standardize=True
        )

    # Time series
    time_series = []
    for scan, conf in zip(fmri, confounds):
        ts = masker.fit_transform(scan, confounds=conf)
        time_series.append(ts)

    if not separate_runs:
        time_series = [np.concatenate(time_series, axis=0)]
        task_regressor = [np.concatenate(task_regressor)]
        run_ids = ["".join(str(ids) for ids in run_ids)]

    if only_task:
        correlation_matrices = connectivity_measure.fit_transform(
            [ts[tr >= 0] for ts, tr in zip(time_series, task_regressor)]
        )
    else:
        correlation_matrices = connectivity_measure.fit_transform(time_series)

    return {
        str(run_ids[i]): correlation_matrices[i] for i in range(len(run_ids))
    }


def run_task_reg_correlation_single(
    path_to_dataset="/users2/local/alix/out2",
    path_to_events="/users2/local/alix/XP2",
    subjects_to_include=["sub-xp201"],
    task_name="NF",
    regression="simple",
    mask_img="/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
    space="MNI152NLin2009cAsym_res-2",
    run_ids=[1, 2, 3],
    atlas="/homes/a19lamou/fmri_data_proc/data/glasser",
    prob=True,
    task=True,
    reg_type="FIR",
    gs=None,
    TR=1,
    only_task=True,
    cropped_mask=True,
    separate_runs=True,
    task_id_event_file='language',
    **conn_args,
):
    FC = {}
    for sub in tqdm(subjects_to_include):
        functional_paths = get_subjects_functional_data(
            path_to_dataset,
            sub,
            task=task_name,
            run_ids=run_ids,
            space=space,
        )
        print(functional_paths)
        if "MI" in task_name:
            events = get_event_file(
                path_to_data=path_to_events, task_name=task_name + run_ids
            )
        elif "NF" in task_name:
            events = get_event_file(path_to_data=path_to_events, task_name="1dNF") if "1dNF" in functional_paths[0] else get_event_file(path_to_data=path_to_events, task_name="2dNF")
        else: 
            events = get_bids_files(path_to_events, sub_label=sub.split('-')[-1], modality_folder='func', file_tag='events', filters=[("run", "01")])[0]
        brain_mask = mask_img
        confs = []
        tasks = []
        for img in functional_paths:
            confounds, task_regressor = load_confounds(
                events,
                img,
                strategy=regression,
                task=task,
                reg_type=reg_type,
                gs=gs,
                TR=TR,
                task_id=task_id_event_file,
            )
            confs.append(confounds)
            tasks.append(task_regressor)
        correlation_matrice = get_fc_roi_single(
            atlas=atlas,
            brain_mask=brain_mask,
            fmri=functional_paths,
            prob=prob,
            confounds=confs,
            task_regressor=tasks,
            only_task=only_task,
            cropped_mask=cropped_mask,
            separate_runs=separate_runs,
            run_ids=run_ids,
            **conn_args,
        )

        FC.update({f"{sub}": correlation_matrice})

    return FC
