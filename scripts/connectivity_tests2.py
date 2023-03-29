import numpy as np
from nilearn import datasets

from fmripreprocessing.connectivity.correlation import (
    run_task_reg_correlation_single,
)
from fmripreprocessing.utils.visualization import (
    plot_glasser_cor_mat,
    plot_HMAT_cor_mat,
)

atlas = datasets.fetch_atlas_difumo(
    dimension=64, resolution_mm=2, legacy_format=False
)

HMAT = "/homes/a19lamou/fmri_data_proc/HMAT/HMAT.nii"
yeo = datasets.fetch_atlas_yeo_2011()
glasser = "/homes/a19lamou/fmri_data_proc/data/Glasser_masker.nii.gz"
scheafer = "/homes/a19lamou/fmri_data_proc/data/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"

subject_list = [
    "sub-xp201",
    "sub-xp202",
    "sub-xp203",
    "sub-xp204",
    "sub-xp205",
    "sub-xp206",
    "sub-xp207",
    "sub-xp210",
    "sub-xp211",
    "sub-xp213",
    "sub-xp216",
    "sub-xp217",
    "sub-xp218",
    "sub-xp220",
    "sub-xp221",
    "sub-xp222",
]

HMA_key = [
    "Right_M1",
    "Left_M1",
    "Right_S1",
    "Left_S1",
    "Right_SMA",
    "Left_SMA",
    "Right_preSMA",
    "Left_preSMA",
    "Right_PMd",
    "Left_PMd",
    "Right_PMv",
    "Left_PMv",
]


atlas_list = {"Scheafer-400": scheafer}
for name, atlas in atlas_list.items():
    print("Running correlation on atlas : ", name)
    results = run_task_reg_correlation_single(
        subjects_to_include=subject_list[:2],
        run_ids=[1, 3],
        atlas=atlas,
        prob=False,
        path_to_dataset="/users2/local/alix/out",
        space="MNI152NLin2009cAsym_res-2",
        kind="correlation",
        task=True,
        gs="basic",
        only_task=True,
    )
    for run in [1, 3]:
        try:
            correlation1 = np.array(
                [corr[run]["corr"] for _, corr in results.items()]
            )
            np.save(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/correlation_{name}_run_{run}_taskReg2.npy",
                correlation1,
            )

        except Exception as ex:
            print(ex)
            continue
atlas_list = {"Scheafer-400": scheafer}
for name, atlas in atlas_list.items():
    print("Running correlation on atlas : ", name)
    results = run_task_reg_correlation_single(
        subjects_to_include=subject_list[:2],
        run_ids=[1, 3],
        atlas=atlas,
        prob=False,
        path_to_dataset="/users2/local/alix/out",
        space="MNI152NLin2009cAsym_res-2",
        kind="correlation",
        task=False,
        gs="basic",
        only_task=True,
    )
    for run in [1, 3]:
        try:
            correlation1 = np.array(
                [corr[run]["corr"] for _, corr in results.items()]
            )
            np.save(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/correlation_{name}_run_{run}_NOtaskReg2.npy",
                correlation1,
            )

        except Exception:
            continue
