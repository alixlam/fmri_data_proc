import matplotlib.pyplot as plt
import nibabel as nib
import nilearn.image as nim
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiSpheresMasker
from nilearn.plotting import plot_anat, plot_roi, plot_stat_map
from tqdm import tqdm

from fmripreprocessing.GLM.first_glm import run_first_level
from nilearn import plotting


def extract_seed_roi(func_data, HMAT, brain_mask=None):
    HMAT_masker = NiftiLabelsMasker(HMAT, mask_img=brain_mask)
    time_serie_ROI = HMAT_masker.fit_transform(func_data)
    return time_serie_ROI

def extract_seed_roi_highest_activity(func_data, HMAT, brain_mask, event_file, task_name, TR):
    contrast_map = run_first_level(event_file, func_data, brain_mask, task_name, TR)
    coords = plotting.find_xyz_cut_coords(contrast_map[task_name], HMAT)
    seed_masker = NiftiSpheresMasker(coords, radius=8, detrend=True, standardize=True)
    time_serie_ROI = seed_masker.fit_transform(func_data)
    return time_serie_ROI

def compute_correlation(func_data, HMAT, brain_mask, event_file, task_name, TR, ROI = True):
    seed_time_serie = extract_seed_roi(func_data, HMAT, brain_mask) if ROI else extract_seed_roi_highest_activity(func_data, HMAT, brain_mask, event_file, task_name, TR)

    brain_masker = NiftiMasker(mask_img=brain_mask, detrend=True, standardize=True)
    brain_time_series = brain_masker.fit_transform(func_data)

    seed_to_voxel_correlations = (np.dot(brain_time_series.T, seed_time_serie) / seed_time_serie.shape[0])

    seed_to_voxel_correlation_img = brain_masker.inverse_transform(seed_to_voxel_correlations.T)
    
    display = plotting.plot_stat_map(seed_to_voxel_correlation_img, title=f"Seed to voxel correlation")
    
    return display, seed_to_voxel_correlations 



    