from fmripreprocessing.connectivity.utils import run_seed_correlation_dataset
from nilearn.masking import intersect_masks
import os 
import pandas as pd
from nilearn.glm.second_level import SecondLevelModel
from fmripreprocessing.GLM.first_glm import run_first_model_dataset
from scipy.stats import norm
import matplotlib.pyplot as plt
from nilearn import plotting
from fmripreprocessing.GLM.second_glm import run_second_model



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

HMAT_full = "/homes/a19lamou/fmri_data_proc/HMAT/HMAT.nii"


results_activity = run_first_model_dataset(path_to_dataset = "/users2/local/alix/out", subjects_to_include= subject_list, regression = "simple",task_name="NF", run_ids = [1,2,3], space = "MNI152NLin2009cAsym_res-2", output_path="/homes/a19lamou/fmri_data_proc/data/glm_simple")

fig, axes = plt.subplots(nrows=5, ncols= 4, figsize=(25,25))
for cidx, (id, tmap) in enumerate(zip(subject_list, data2)):
    plotting.plot_glass_brain(
        tmap,
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
import os 
data2 = [os.path.join(f"/homes/a19lamou/fmri_data_proc/data/glm/{sub}_cont-TaskNF - Rest_fist_level.nii.gz") for sub in subject_list]

result2 = run_second_model(data2)

display = plotting.plot_stat_map(
        result2[0],
        threshold=norm.isf(0.001),
        colorbar=True,
        display_mode="ortho",
        title="[TaskNF - Rest] (uncor < 0.001)",
    )
display.add_contours(HMAT_full)


# Thresholds
from nilearn.glm import threshold_stats_img
thresholded_bonf, threshold1 = threshold_stats_img(
    result2[0],
    alpha=.05,
    height_control='bonferroni',
    cluster_threshold=0,
    two_sided=True,
)
thresholded_fdr, threshold2 = threshold_stats_img(
    result2[0],
    alpha=.05,
    height_control='fdr',
    cluster_threshold=10,
    two_sided=True,
)


display = plotting.plot_glass_brain(
            thresholded_bonf,
            threshold=threshold1,
            colorbar=True,
            display_mode="lzry",
            plot_abs=False,
            title="[TaskNF - Rest] Group maps (p < 0.05 Bonferroni)",
        )
display.add_contours(HMAT_full)

display = plotting.plot_stat_map(
            thresholded_fdr,
            threshold=threshold2,
            colorbar=True,
            display_mode="ortho",
            title="[TaskNF - Rest] Group maps (p < 0.05 FDR)",
        )
display.add_contours(HMAT_full)

display = plotting.plot_stat_map(
            thresholded_bonf,
            threshold=threshold1,
            colorbar=True,
            display_mode="ortho",
            title="[TaskNF - Rest] Group maps (p < 0.05 FDR)",
        )
display.add_contours(HMAT_full)

# TFCE
from nilearn.glm.second_level import non_parametric_inference
design_matrix = pd.DataFrame([1] * len(data2), columns=["intercept"])
out_dict = non_parametric_inference(
    data2,
    design_matrix=design_matrix,
    model_intercept=False,
    n_perm=1000,  # 500 for the sake of time. Ideally, this should be 10,000.
    two_sided_test=False,
    smoothing_fwhm=6.0,
    n_jobs=5,
    threshold=0.001,
    tfce = True , 
    verbose = 1,
)

display = plotting.plot_glass_brain(
        out_dict["logp_max_tfce"],
        threshold=1.3,
        colorbar=True,
        display_mode="lzr",
        plot_abs=False,
        title="[TaskNF - Rest] Group maps (p < 0.05 TFCE FWE)",
    )
display.add_contours(HMAT_full)

display = plotting.plot_glass_brain(
        out_dict["logp_max_t"],
        threshold=1.3,
        colorbar=True,
        display_mode="lzry",
        plot_abs=False,
        title="[TaskNF - Rest] Group maps (p < 0.05 FWER bonferroni)",
    )
display.add_contours(HMAT_full)

display = plotting.plot_glass_brain(
        out_dict["logp_max_mass"],
        threshold=1.3,
        colorbar=True,
        display_mode="lzry",
        plot_abs=False,
        title="[TaskNF - Rest] Group maps (p < 0.05 FWER bonferroni)",
    )
display.add_contours(HMAT_full)