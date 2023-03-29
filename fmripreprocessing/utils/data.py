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


def get_subject_NF_run(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub-xp201",
    run_ids=[1, 2, 3],
    space="MNI152NLin2009cAsym",
):
    files = []
    for id in run_ids:
        files += glob.glob(
            os.path.join(
                path_to_data,
                subject_id,
                "func",
                f"{subject_id}_task-*dNF_run-{id}_space-{space}_desc-preproc_bold.nii.gz",
            )
        )
    return files


def get_subject_brain_mask_from_T1(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub_xp201",
    space="MNI152NLin2009cAsym",
    GM=True,
):
    T1_brain_mask = os.path.join(
        path_to_data,
        subject_id,
        "anat",
        f"{subject_id}_space-{space}_desc-brain_mask.nii.gz",
    )
    bold_example = os.path.join(
        path_to_data,
        subject_id,
        "func",
        f"{subject_id}_task-MIpre_space-{space}_desc-preproc_bold.nii.gz",
    )
    mask_fov = compute_epi_mask(bold_example)
    T1_bold = resample_mask_to_bold(bold_example, T1_brain_mask)
    GM_mask_bold = resample_mask_to_bold(bold_example, load_mni152_gm_mask())
    if GM:
        final = intersect_masks([T1_bold, mask_fov, GM_mask_bold], 1)
    else:
        final = intersect_masks([T1_bold, mask_fov], 1)
    return final


def get_subject_MI(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub-xp201",
    run_ids="pre",
    space="MNI152NLin2009cAsym",
):
    return [
        os.path.join(
            path_to_data,
            subject_id,
            "func",
            f"{subject_id}_task-MI{run_ids}_space-{space}_desc-preproc_bold.nii.gz",
        )
    ]


def get_subjects_functional_data(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub-xp201",
    space="MNI152NLin2009cAsym",
    task="NF",
    run_ids=[1, 2, 3],
):
    
    if "NF" in task:
        return get_subject_NF_run(path_to_data, subject_id, run_ids, space)
    elif "MI" in task:
        return get_subject_MI(path_to_data, subject_id, run_ids, space)
    else:
        filters = [] 
        if space is not None:
            filters.append(("space", space))
        
        return get_bids_files(path_to_data, file_tag='bold', file_type='nii*',
                                    sub_label=subject_id.split('-')[-1],
                                    modality_folder='func', filters=filters)
            


def get_event_file(path_to_data="/users2/local/alix/XP2", task_name="1dNF"):
    return os.path.join(path_to_data, f"task-{task_name}_events.tsv")


def fetch_difumo_fov(
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
