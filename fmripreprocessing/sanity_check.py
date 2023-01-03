from nilearn.plotting import plot_anat, plot_stat_map, plot_roi
import matplotlib.pyplot as plt
import nilearn.image as nim
from nilearn.maskers import NiftiMasker, NiftiLabelsMasker
import pandas as pd 
from tqdm import tqdm
import nibabel as nib
import numpy as np

def plot_func_segm(brain_mask, img, anat):
    masker = NiftiMasker(brain_mask)
    masker.fit(img)
    mean_func = nim.mean_img(img)

    plot_roi(brain_mask, anat,view_type="contour" )


def extract_seed_roi(func_data, HMAT, brain_mask=None):
    HMAT_masker = NiftiLabelsMasker(HMAT, mask_img=brain_mask)
    time_serie_ROI = HMAT_masker.fit_transform(func_data)
    return time_serie_ROI


def extract_task_events(imgs, events_file):
    """
    Extract volumes corresponding to events and concatenate them.

    Parameters
    ----------
    imgs : str or Nifti1Image
        path to fMRI volume or nifti image corresponding to volume (4D image)
    events_file : str
        path to events files.

    Returns
    -------
    Nifti1Image  
        Concatenate task events fMRI
    """
    events = pd.read_table(events_file)
    task_events = events[events["trial_type"] != "Rest"]
    if type(imgs) == 'str': 
        img = nib.load(imgs)
    else:
        img = imgs
    affine = img.affine
    img = img.get_fdata()
    img_events = []
    for onset, duration in tqdm(zip(task_events["onset"], task_events["duration"])):
        img_events.append(img[:,:,:,onset : onset + duration])
    return nib.Nifti1Image(np.concatenate(img_events, axis=3), affine)
