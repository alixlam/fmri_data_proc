import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from fmripreprocessing.connectivity.confounds import (
    get_task_function,
    load_confounds,
)
from fmripreprocessing.utils.data import *
from fmripreprocessing.utils.masks import intersect_multilabel


class PPI:
    def __init__(self, tr=1) -> None:
        self.tr = 1
        self.results = None
        self.task_regressor = None

    def fit_transform(self, time_series, events=None, dm=None):
        ppi_coefficients = []

        if events is not None:
            frame_times = np.arange(len(time_series) * self.tr) + self.tr / 2
            task = get_task_function(events=events, frame_times=frame_times)
            dm = pd.DataFrame(
                {"Task": task, "constant": np.ones(len(frame_times))}
            )

        for roi in time_series.T:
            dm["roi"] = roi
            psy = dm.iloc[:, 0]
            psy = psy - np.mean([psy.min(), psy.max()])
            dm["ppi"] = psy * dm["roi"]
            model = LinearRegression(fit_intercept=False, n_jobs=-2)
            model.fit(dm, time_series)
            ppi_coefficients.append(model.coef_[:, -1])
        return np.vstack(ppi_coefficients)


def get_ppi_single(
    atlas,
    brain_mask,
    fmri,
    prob=False,
    confounds=None,
    cropped_mask=True,
    events=None,
    tr=1,
    separate_runs=True,
    run_ids=[1, 2, 3],
):
    ppi = PPI(tr=tr)
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
    time_series = []
    for scan, conf in zip(fmri, confounds):
        ts = masker.fit_transform(scan, confounds=conf)
        time_series.append(ts)

    if not separate_runs:
        time_series = [np.concatenate(time_series, axis=0)]
        run_ids = ["".join(str(ids) for ids in run_ids)]

    runs = {}
    for i, ts in enumerate(time_series):
        if not separate_runs:
            frame_times = np.arange(len(ts) * tr) + tr / 2
            task = get_task_function(events=events, frame_times=frame_times)
            task_all_ts = [task for _ in range(len(run_ids))]
            task_final = np.concatenate(task_all_ts)
            dm = pd.DataFrame(
                {"Task": task_final, "constant": np.ones(len(task_final))}
            )
            ppi_conectivity = ppi.fit_transform(ts, dm=dm)
        else:
            ppi_conectivity = ppi.fit_transform(ts, dm=dm)
        runs.update({str(run_ids[i]): ppi_conectivity})

    return runs


def run_ppi(
    path_to_dataset="/users2/local/alix/out",
    path_to_events="/users2/local/alix/XP2",
    subjects_to_include=["sub-xp201"],
    task_name="NF",
    regression="simple",
    mask_img="/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
    space="MNI152NLin2009cAsym_res-2",
    output_path="/homes/a19lamou/fmri_data_proc/data/correlation",
    run_ids=[1, 2, 3],
    atlas="/homes/a19lamou/fmri_data_proc/data/glasser",
    prob=False,
    task=False,
    gs=None,
    TR=1,
    cropped_mask=True,
    separate_runs=True,
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
        brain_mask = mask_img
        confs = []
        for img in functional_paths:
            confounds, __ = load_confounds(
                events, img, strategy=regression, task=task, gs=gs, TR=TR
            )
            confs.append(confounds)
        ppi = get_ppi_single(
            atlas=atlas,
            brain_mask=brain_mask,
            fmri=functional_paths,
            prob=prob,
            confounds=confs,
            cropped_mask=cropped_mask,
            events=events,
            separate_runs=separate_runs,
            run_ids=run_ids,
        )
        FC.update({f"{sub}": ppi})

    return FC
