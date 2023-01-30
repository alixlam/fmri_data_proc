from fmripreprocessing.connectivity.utils import run_seed_correlation_dataset
from nilearn.masking import intersect_masks
import os 
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from fmripreprocessing.GLM.first_glm import run_first_model_dataset
from scipy.stats import norm
import matplotlib.pyplot as plt


subject_list = ['sub-xp201',
 'sub-xp202',
 'sub-xp203',
 'sub-xp204',
 'sub-xp205',
 'sub-xp206',
 'sub-xp207',
 'sub-xp210',
 'sub-xp211',
 'sub-xp213',
 'sub-xp216',
 'sub-xp217',
 'sub-xp218',
 'sub-xp219',
 'sub-xp220',
 'sub-xp221',
 'sub-xp222']

HMAT = intersect_masks(["/homes/a19lamou/fmri_data_proc/HMAT/HMAT_Left_PMd.nii", "/homes/a19lamou/fmri_data_proc/HMAT/HMAT_Left_PMv.nii"], 0)

correlation_results, _= run_seed_correlation_dataset(path_to_dataset = "/users2/local/alix/out",space = "MNI152NLin2009cAsym_res-2",subjects_to_include=subject_list,HMAT=HMAT)

from nilearn import plotting
import matplotlib.pyplot as plt
HMAT_full = "/homes/a19lamou/fmri_data_proc/HMAT/HMAT.nii"
fig, axes = plt.subplots(nrows=5, ncols= 4, figsize=(25,25))
for cidx, (id, tmap) in enumerate(correlation_results.items()):
    plotting.plot_glass_brain(
        tmap,
        colorbar=True,
        title=id,
        axes=axes[int(cidx / 4), int(cidx % 4)],
        plot_abs=False,
        display_mode='z',
        black_bg=False
    ).add_contours(HMAT_full)
fig.suptitle('PMC seed connectivity')

from fmripreprocessing.GLM.second_glm import run_second_model

def run_second_model(data, plot=True, p_val_thresh=0.001):
    design_matrix = pd.DataFrame([1] * len(data), columns=["intercept"])
    second_level_glm = SecondLevelModel(smoothing_fwhm=4)
    second_level_glm.fit(data, design_matrix=design_matrix)

    contrast = second_level_glm.compute_contrast(
        second_level_contrast="intercept",
        output_type="z_score",
    )

    if plot:
        display = plotting.plot_glass_brain(
            contrast,
            threshold=norm.isf(p_val_thresh),
            colorbar=True,
            display_mode="lzry",
            plot_abs=False,
            title="[TaskNF - Rest] Group maps",
        )

    return contrast, second_level_glm
data = [dat for i, dat in correlation_results.items()]
result = run_second_model(data)

display = plotting.plot_stat_map(
        result[0],
        threshold=norm.isf(0.001),
        colorbar=True,
        display_mode="mosaic",
        title="Group effect correlation (uncor - 0.001)",
    )
display.add_contours(HMAT_full)


results_activity = run_first_model_dataset(path_to_dataset = "/users2/local/alix/out", subjects_to_include= subject_list, task_name="NF", run_ids = [1,2,3], space = "MNI152NLin2009cAsym_res-2")

fig, axes = plt.subplots(nrows=5, ncols= 4, figsize=(25,25))
for cidx, (id, tmap) in enumerate(results_activity.items()):
    plotting.plot_glass_brain(
        tmap["TaskNF - Rest"],
        colorbar=True,
        threshold=2.0,
        title=id,
        axes=axes[int(cidx / 4), int(cidx % 4)],
        plot_abs=False,
        display_mode='z',
        black_bg=False
    ).add_contours(HMAT_full)
fig.suptitle('Subjects t_maps NF - Rest')

data2 = [dat["TaskNF - Rest"] for i, dat in results_activity.items()]

result2 = run_second_model(data2)

display = plotting.plot_stat_map(
        result2[0],
        threshold=norm.isf(0.001),
        colorbar=True,
        display_mode="mosaic",
        title="[TaskNF - Rest] (uncor < 0.001)",
    )
display.add_contours(HMAT_full)


results_activity = run_first_model_dataset(path_to_dataset = "/users2/local/alix/out", subjects_to_include= subject_list, task_name="NF", run_ids = [1,2,3], space = "MNI152NLin2009cAsym_res-2", regression=None)

