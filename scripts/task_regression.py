import sys

import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn.glm.first_level import FirstLevelModel

from fmripreprocessing.connectivity.confounds import load_confounds
from fmripreprocessing.GLM.second_glm import run_second_model
from fmripreprocessing.utils.data import *

sys.path.append("../")

subject_list_1d = [
    "sub-xp201",
    "sub-xp202",
    "sub-xp203",
    "sub-xp206",
    "sub-xp210",
    "sub-xp211",
    "sub-xp217",
    "sub-xp218",
    "sub-xp219",
    "sub-xp220",
    "sub-xp222",
]

subject_list_2d = [
    "sub-xp204",
    "sub-xp205",
    "sub-xp207",
    "sub-xp210",
    "sub-xp213",
    "sub-xp216",
    "sub-xp221",
]

betas = []
events_id = [0] * len(subject_list_1d) + [1] * len(subject_list_2d)
events_file = [
    get_event_file(task_name="1dNF"),
    get_event_file(task_name="2dNF"),
]
for sub, event in zip(subject_list_1d + subject_list_2d, events_id):
    fmri = get_subjects_functional_data(
        path_to_data="/users2/local/alix/out",
        subject_id=sub,
        space="MNI152NLin2009cAsym_res-2",
        run_ids=[1],
    )[0]
    dm, _ = load_confounds(fmri_path=fmri, events=events_file[event])
    glm = FirstLevelModel(
        t_r=1,
        mask_img="/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
    )
    glm.fit(fmri, design_matrices=[dm])
    betas.append(glm.compute_contrast("Task", output_type="stat"))

fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25, 25))
for cidx, (id, tmap) in enumerate(
    zip(subject_list_1d + subject_list_2d, betas)
):
    plotting.plot_glass_brain(
        tmap,
        colorbar=True,
        threshold=2.0,
        title=id,
        axes=axes[int(cidx / 4), int(cidx % 4)],
        plot_abs=False,
        display_mode="ortho",
        black_bg=False,
    )
fig.suptitle("Subjects t_maps NF - Rest")
