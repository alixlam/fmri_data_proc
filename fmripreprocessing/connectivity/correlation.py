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

    return {
        str(run_ids[i]): correlation_matrices[i] for i in range(len(run_ids))
    }

def get_task_FC(
    path_to_dataset="/users2/local/alix/out2",
    path_to_events="/users2/local/alix/XP2",
    subjects_to_include=["sub-xp201"],
    task_name="NF",
    denoise_strategy="simple",
    brain_mask="/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
    space="MNI152NLin2009cAsym_res-2",
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
                #cleaned_fmri = functional_paths[0]
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
