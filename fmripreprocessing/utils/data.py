import glob
import os

import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.datasets import load_mni152_gm_mask
from nilearn.masking import compute_epi_mask, intersect_masks
from nilearn.interfaces.bids import get_bids_files

from fmripreprocessing.utils.masks import (
    intersect_multilabel,
    resample_mask_to_bold,
)
from fmripreprocessing.utils.lesion_masks import save_lesion_masks_derivatives

def get_subject_functional_data(
    path_to_data='/users2/local/alix/out',
    subject_id="sub-xp201",
    run_id = 1,
    ses_id=None,
    space="MNI152NLin2009cAsym",
    task_name="1dNF",
):
    filters = []
    if space is not None:
        filters.append(("space", space))
    if ses_id is not None:
        filters.append(("ses", ses_id))
    if task_name is not None:
        filters.append(("task", task_name))
    if run_id is not None:
        filters.append(("run", str(run_id)))
    subject_id = subject_id.split('-')[-1]
    return get_bids_files(path_to_data, sub_label = subject_id, file_tag='bold', file_type='nii*', 
                            modality_folder='func', filters=filters)

def get_event_file(path_to_events="/users2/local/alix/XP2", task_name="1dNF", subject_id="*", sub_folder=True):
    return get_bids_files(path_to_events, sub_label=subject_id.split("-")[-1], file_tag="events", filters=[("task", task_name)], sub_folder = sub_folder)
    


def get_subject_brain_mask_from_T1(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub_xp201",
    space="MNI152NLin2009cAsym",
    GM=True,
):
    if "T1" in space:
        T1_brain_mask = os.path.join(
            path_to_data,
            subject_id,
            "anat",
            f"{subject_id}_desc-brain_mask.nii.gz",
        )
    else:
        T1_brain_mask = os.path.join(
            path_to_data,
            subject_id,
            "anat",
            f"{subject_id}_space-{space}_desc-brain_mask.nii.gz",
        )
    bold_example = glob.glob(os.path.join(path_to_data, subject_id, "**", f"*space-{space}*desc-preproc_bold.nii.gz"), recursive=True)[0]
    """os.path.join(
        path_to_data,
        subject_id,
        "func",
        f"{subject_id}_task-MIpre_space-{space}_desc-preproc_bold.nii.gz",
    )"""
    mask_fov = compute_epi_mask(bold_example)
    T1_bold = resample_mask_to_bold(bold_example, T1_brain_mask)
    if GM:
        if "MNI" in space:
            GM = load_mni152_gm_mask()
        elif "T1" in space:
            GM_tmp = nib.load(os.path.join(os.path.dirname(T1_brain_mask), f"{subject_id}_dseg.nii.gz"))
            GM_tmp_array = GM_tmp.get_fdata()
            GM_tmp_array[GM_tmp_array != 1] = 0
            GM = nib.Nifti1Image(GM_tmp_array, affine=GM_tmp.affine)
        GM_mask_bold = resample_mask_to_bold(bold_example, GM)
        final = intersect_masks([T1_bold, mask_fov, GM_mask_bold], 1)
    else:
        final = intersect_masks([T1_bold, mask_fov], 1)
    return final


"""def fetch_difumo_fov(
    fov_mask="/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
    dimension=64,
):
    atlas = datasets.fetch_atlas_difumo(dimension=dimension)
    atlas_fov = intersect_multilabel(fov_mask=fov_mask, labels_img=atlas.maps)
    atlas_fov_array = atlas_fov.get_fdata()
    indices_in_fov = [
        i
        for i in range(atlas_fov_array.shape[-1])
        if np.sum(atlas_fov_array[:, :, :, i] != 0)
    ]
    final_atlas_fov_array = np.zeros(
        (
            atlas_fov_array.shape[0],
            atlas_fov_array.shape[1],
            atlas_fov_array.shape[2],
            len(indices_in_fov),
        )
    )
    for new_i, i in enumerate(indices_in_fov):
        final_atlas_fov_array[:, :, :, new_i] = atlas_fov_array[:, :, :, i]
    final_atlas_fov = nib.Nifti1Image(final_atlas_fov_array, atlas_fov.affine)
    labels_left = atlas.labels[indices_in_fov]
    labels_left["component"]
    return {"maps": final_atlas_fov, "labels": labels_left}
"""

def check_data(path_to_data):
    subjects = glob.glob(os.path.join(path_to_data, "sub*"))
    def allEqual(iterable):
        iterator = iter(iterable)
    
        try:
            firstItem = next(iterator)
        except StopIteration:
            return True
            
        for x in iterator:
            if x!=firstItem:
                return False
        return True
    T1w_test = {}
    sub_t1_prob = []
    sub_lesion_prob = []
    for sub in subjects:
        T1 = True
        check_dim = True
        subject_id = sub.split("/")[-1]
        T1w = glob.glob(os.path.join(sub ,"**", "anat", "*T1w*"), recursive=True)
        shapesT1w = []
        for file in T1w:
            shapesT1w.append(nib.load(file).shape)
        if not allEqual(shapesT1w):
            T1 = False
            T1w_test.update({sub: {"T1": T1w, "shapes": shapesT1w}})
            sub_t1_prob.append(subject_id)
        if os.path.isdir(os.path.join(sub, "anat")):
            lesions_file = os.path.join(sub, "anat", f"{subject_id}_T1w.nii.gz"),  os.path.join(sub, "anat", f"{subject_id}_lesion_roi.nii.gz")
            check_dim = nib.load(lesions_file[0]).shape == nib.load(lesions_file[1]).shape
            if not check_dim:
                sub_lesion_prob.append(subject_id)
        
    if len(sub_lesion_prob) == 0 and len(sub_t1_prob) == 0:
        print("youppii")
        return True
    if len(sub_t1_prob) != 0:
        print("problems with subjects : \n", sub_t1_prob)
    if len(sub_lesion_prob) != 0:
        print("problem with lesions of : \n", sub_lesion_prob)
        
    return False

def get_lagmaps_inputs(subjects_id, pathdata_derivatives, space='MNI152NLin2009cAsym', pathdata = None):
    
    # Lesion ROI in Bold
    # Check if already exist 
    saving_path_lesion = os.path.join(pathdata_derivatives, subjects_id, "masks", f"{subjects_id}_space-{space}_lesion_roi_bold.nii.gz")
    path_to_bold = glob.glob(os.path.join(pathdata_derivatives, subjects_id, "*", "func", f"{subjects_id}_*_space-{space}_desc-preproc_bold.nii.gz"))
    if not os.path.isfile(saving_path_lesion):
        path_to_lesion = os.path.join(pathdata_derivatives, subjects_id, "anat", f"{subjects_id}_space-{space}_lesion_roi.nii.gz")
        if not os.path.isfile(path_to_lesion):
            pathdata = pathdata_derivatives.replace("_derivatives", "")
            save_lesion_masks_derivatives(subjects_id, pathdata, pathdata_derivatives, pathout = None)
        lesion_roi_bold = resample_mask_to_bold(bold_img=path_to_bold[0], mask=path_to_lesion)
        try:
            nib.save(lesion_roi_bold, saving_path_lesion)
        except:
            os.mkdir(os.path.dirname(saving_path_lesion))
            nib.save(lesion_roi_bold, saving_path_lesion)
    saving_path_GM = os.path.join(pathdata_derivatives, subjects_id, "masks", f"{subjects_id}_space-{space}_GM.nii.gz")
    saving_path_fov = os.path.join(pathdata_derivatives, subjects_id, "masks", f"{subjects_id}_space-{space}_fov.nii.gz")
    if not os.path.isfile(saving_path_GM):
        GM_mask = get_subject_brain_mask_from_T1(path_to_data=pathdata_derivatives, subject_id=subjects_id, space=space, GM=True)
        nib.save(GM_mask, saving_path_GM)
    if not os.path.isfile(saving_path_fov):
        FOV_mask = get_subject_brain_mask_from_T1(path_to_data=pathdata_derivatives, subject_id=subjects_id, space=space, GM=False)
        nib.save(FOV_mask, saving_path_fov)
    path_to_confounds = path_to_bold[0].replace(f"space-{space}_desc-preproc_bold.nii.gz", "desc-confounds_timeseries.tsv")
    return saving_path_lesion, saving_path_GM, saving_path_fov, path_to_bold[0], path_to_confounds

def fetch_sub_fsaverage(pathdata_derivatives, subject):
    path = os.path.join(pathdata_derivatives,"sourcedata", "freesurfer",subject)
    sub_surf = {}
    sub_atlas = {}
    for hemi, hemifile in zip(["left", "right"], ["lh", "rh"]):
        sub_surf.update({
            f"infl_{hemi}": os.path.join(path, "surf", f"{hemifile}.inflated"),
            f"sulc_{hemi}": os.path.join(path, "surf", f"{hemifile}.sulc"), 
            f"pial_{hemi}": os.path.join(path, "surf", f"{hemifile}.pial"), 
        })
        sub_atlas.update({
            f"atlas_{hemi}": os.path.join(path, "label", f"{hemifile}.aparc.a2009s.annot")
        })
    return sub_surf, sub_atlas