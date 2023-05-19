import nibabel as nib
import nilearn.image as nim
import numpy as np
import pandas as pd
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from tqdm import tqdm

from fmripreprocessing.utils.data import *
from fmripreprocessing.utils.masks import intersect_multilabel

from fmripreprocessing.utils.confounds import *

def get_fc_roi(
    atlas,
    brain_mask,
    fmri,
    prob=True,
    cropped_mask=True,
    separate_runs=True,
    run_ids=[1, 2, 3],
    return_masker = False,
    **conn_args,
):
    connectivity_measure = ConnectivityMeasure(**conn_args)
    if cropped_mask:
        mask_cropped = atlas
    else:
        mask_cropped = intersect_multilabel(brain_mask, atlas)
    if not prob:
        masker = NiftiLabelsMasker(
            mask_cropped, mask_img=brain_mask, standardize=False, detrend= False, standardize_confounds = False, verbose=0
        )

    else:
        masker = NiftiMapsMasker(
            atlas.maps, mask_img=brain_mask, standardize=False
        )

    # Time series
    time_series = []
    for scan in fmri:
        ts = masker.fit_transform(scan)
        time_series.append(ts)

    if not separate_runs:
        time_series = [np.concatenate(time_series, axis=0)]
        run_ids = ["".join(str(ids) for ids in run_ids)]

    correlation_matrices = connectivity_measure.fit_transform(time_series)
    if return_masker:
        return {
            str(run_ids[i]): correlation_matrices[i] for i in range(len(run_ids))
        }, masker
    else:
        return {
        str(run_ids[i]): correlation_matrices[i] for i in range(len(run_ids))
    }
def get_fc_homotopic(
    atlas,
    brain_mask,
    fmri,
    atlas_labels,
    atlas_labels_name,
    cropped_mask = True,
    run_ids= [1,2,3],
    **conn_args,

):
    
    corr_mat, masker = get_fc_roi(atlas = atlas, brain_mask = brain_mask, fmri = fmri, cropped_mask = cropped_mask, prob=False, run_ids = run_ids, separate_runs = True, return_masker=True,    **conn_args,
)
    LH_labels = [lab for lab in atlas_labels_name if '_LH_' in lab]
    RH_labels = [lab for lab in atlas_labels_name if '_RH_' in lab]
    LH_labels = [label for label in LH_labels if label.replace('LH', 'RH') in RH_labels]
    print(masker.n_elements_)

    homotopic_FC_maps = {}
    for run, cor in corr_mat.items():
        homotopic_FC = np.zeros(nib.load(atlas).shape)
        for region in LH_labels:
            i = atlas_labels_name.index(region)
            j = atlas_labels_name.index(region.replace("LH", "RH"))
            homotopic_FC[np.where(nib.load(atlas).get_fdata() == atlas_labels[i])] = cor[i,j]
            homotopic_FC[np.where(nib.load(atlas).get_fdata() == atlas_labels[j])] = cor[i,j]
        homotopic_map = nib.Nifti1Image(homotopic_FC, affine = nib.load(atlas).affine)
        homotopic_FC_maps.update({str(run): homotopic_map})   
    return homotopic_FC_maps 


def get_task_FC(
    path_to_dataset="/users2/local/alix/out2",
    path_to_events="/users2/local/alix/XP2",
    subjects_to_include=["sub-xp201"],
    task_name="NF",
    denoise_strategy="simple",
    brain_mask="/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
    space="MNI152NLin2009cAsym",
    run_ids=[1, 2, 3],
    ses_id=[None],
    atlas="/homes/a19lamou/fmri_data_proc/data/masks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    prob=False,
    task=True,
    reg_type="FIR",
    firlag=None, 
    gs=None,
    TR=None,
    task_block=True,
    cropped_mask=True,
    separate_runs=True,
    task_id_event_file='language',
    xp2 = False,
    verbose=True,
    **conn_args,
):
    FC = {}
    for sub in tqdm(subjects_to_include):
        if xp2 and "NF" in task_name:
            table = pd.read_table(os.path.join(path_to_events, "participants.tsv"))
            task_name = table[table["participant_id"] == sub]["feedback_type"].iloc[0]+"NF"
        FCses = {}
        for ses in ses_id:
            denoised_functional_img = []
            for run in run_ids:
                functional_paths = get_subject_functional_data(
                    path_to_dataset,
                    sub,
                    task_name=task_name,
                    run_id=run,
                    space=space,
                    ses_id=ses,
                )
                if verbose:
                    print(f"Loading : {functional_paths}")

                events = get_event_file(path_to_events=path_to_events, task_name=task_name, subject_id=sub)
                if len(events) == 0:
                    if verbose :
                        print("No event found in subject folder, looking in dataset folder")
                    events = get_event_file(path_to_events=path_to_events, task_name=task_name, subject_id="*", sub_folder=False)
                if len(events) > 1:
                    events = [ev for ev in events if f"run-{run}" in ev]
                if verbose:
                    print(f"loading events file : {events}")
                
                # Step 1 : Remove confounds:
                cleaned_fmri = confound_regression(func_img=functional_paths[0], confound_strategy=denoise_strategy, gs=gs, mask_img=brain_mask)

                # Step 1 bis : remove task 
                if task: 
                    cleaned_fmri = task_regression(cleaned_fmri, events[0], regression_strategy=reg_type, firlag=firlag, task_id=task_id_event_file, mask_img=brain_mask)

                # Step 2 : Extract time series
                if task_block:
                    task_function = get_task_function(events[0], TR=TR, task_id=task_id_event_file, ntime=cleaned_fmri.shape[-1])
                    new_fmri_task_blocks_array = cleaned_fmri.get_data()[:,:,:, task_function > 0]
                    cleaned_fmri_final = nib.Nifti1Image(new_fmri_task_blocks_array, affine=cleaned_fmri.affine)
                else :
                    cleaned_fmri_final = cleaned_fmri
                denoised_functional_img.append(cleaned_fmri_final)
            
            connectivity_matrice = get_fc_roi(
                atlas=atlas,
                brain_mask=brain_mask,
                fmri=denoised_functional_img,
                prob=prob,
                cropped_mask=cropped_mask,
                separate_runs=separate_runs,
                run_ids=run_ids,
                **conn_args,
            )
            FCses.update({f"{ses}": connectivity_matrice})

        FC.update({f"{sub}": FCses})

    return FC

def get_homotopic_FC_subject(
    path_to_dataset="/users2/local/alix/out2",
    subjects_to_include=["sub-xp201"],
    task_name="NF",
    denoise_strategy="simple",
    brain_mask="/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
    space="MNI152NLin2009cAsym_res-2",
    run_ids=[1, 2, 3],
    ses_id=[None],
    atlas="/homes/a19lamou/fmri_data_proc/data/masks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz",
    atlas_labels_name = None,
    atlas_labels=None,
    gs=None,
    TR=None,    
    cropped_mask=True,
    verbose=True,
    **conn_args,
):
    FC = {}
    for sub in tqdm(subjects_to_include):
        FCses = {}
        for ses in ses_id:
            denoised_functional_img = []
            for run in run_ids:
                functional_paths = get_subject_functional_data(
                    path_to_dataset,
                    sub,
                    task_name=task_name,
                    run_id=run,
                    space=space,
                    ses_id=ses,
                )
                if verbose:
                    print(f"Loading : {functional_paths}")
                
                # Step 1 : Remove confounds:
                if verbose:
                    print("Running temporal denoising ...")
                cleaned_fmri = confound_regression(func_img=functional_paths[0], confound_strategy=denoise_strategy, gs=gs, mask_img=brain_mask)
                
                denoised_functional_img.append(cleaned_fmri)
            
            homotopic_FC = get_fc_homotopic(
                atlas=atlas,
                brain_mask=brain_mask,
                fmri=denoised_functional_img,
                atlas_labels=atlas_labels,
                atlas_labels_name=atlas_labels_name,
                cropped_mask=cropped_mask,
                run_ids=run_ids,
                **conn_args,
            )
            FCses.update({f"{ses}": homotopic_FC})

        FC.update({f"{sub}": FCses})

    return FC