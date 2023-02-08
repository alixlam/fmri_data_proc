import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.image as nim
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker
from nilearn.plotting import plot_anat, plot_roi, plot_stat_map

from fmripreprocessing.GLM.first_glm import run_first_level
from fmripreprocessing.sanity_check import extract_task_events

from fmripreprocessing.utils.data import *
from nilearn.interfaces.fmriprep import load_confounds, load_confounds_strategy


def extract_seed_roi(func_data, HMAT, brain_mask=None):
    HMAT_masker = NiftiLabelsMasker(
        HMAT, mask_img=brain_mask, detrend=False, standardize=True
    )
    time_serie_ROI = HMAT_masker.fit_transform(func_data)
    return time_serie_ROI


def extract_seed_roi_highest_activity(
    func_data, HMAT, brain_mask, event_file, task_name, TR
):
    contrast_map = run_first_level(
        event_file, func_data, brain_mask, task_name, TR
    )
    coords = plotting.find_xyz_cut_coords(contrast_map[task_name], HMAT)
    coords = [tuple(coords)]
    seed_masker = NiftiSpheresMasker(
        coords, radius=8, detrend=True, standardize=True
    )
    time_serie_ROI = seed_masker.fit_transform(func_data)
    return time_serie_ROI


def compute_correlation(
    func_data,
    HMAT,
    brain_mask,
    event_file,
    task_name,
    TR,
    ROI=True,
    title="Seed to voxel correlation",
    value="z-score",
    plot=True,
):

    assert value in [
        "z-score",
        "correlation",
    ], f"Only 'correlation' and 'z-score' are supported for argument 'value', instead got {value}"
    task_time_series = func_data  # extract_task_events(func_data, event_file)
    seed_time_serie = (
        extract_seed_roi(task_time_series, HMAT, brain_mask)
        if ROI
        else extract_seed_roi_highest_activity(
            func_data, HMAT, brain_mask, event_file, task_name, TR
        )
    )
    brain_masker = NiftiMasker(
        mask_img=brain_mask, detrend=False, standardize=True, smoothing_fwhm=4
    )
    brain_time_series = brain_masker.fit_transform(task_time_series)

    seed_to_voxel_correlations = (
        np.dot(brain_time_series.T, seed_time_serie) / seed_time_serie.shape[0]
    )

    if value == "z-score":
        seed_to_voxel_correlations = np.arctanh(seed_to_voxel_correlations)
    else:
        pass

    seed_to_voxel_correlation_img = brain_masker.inverse_transform(
        seed_to_voxel_correlations.T
    )

    if plot:
        display = plotting.plot_stat_map(
            seed_to_voxel_correlation_img, title=title
        )
        display.add_contours(HMAT)

    return seed_to_voxel_correlations, seed_to_voxel_correlation_img


def run_seed_correlation_dataset(
    path_to_dataset="/users2/local/alix/out2",
    path_to_events="/users2/local/alix/XP2",
    subjects_to_include=["sub-xp201"],
    task_name="NF",
    regression="simple",
    run_ids=[1],
    space = "MNI152NLin2009Asym",
    HMAT = None):

    correlation_maps = {}

    for sub in subjects_to_include:
        functional_paths = get_subjects_functional_data(
            path_to_dataset, sub, task=task_name, run_ids=run_ids, space = space,
        )
        if "MI" in task_name:
            events = get_event_file(
                path_to_data=path_to_events, task_name=task_name + run_ids,
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
                functional_paths, regression, global_signal = 'basic', demean = False,
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

        print(clean_img)
        correlations, correlation_map = compute_correlation(func_data = clean_img[0], HMAT=HMAT, brain_mask=brain_mask, event_file=events, task_name=task_name, TR=1, ROI=True, plot=False)

        correlation_maps.update({f"{sub}": correlation_map})
    return correlation_maps, clean_img