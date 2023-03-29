import sys

import matplotlib.pyplot as plt
from configs.config import CONFIG
from nilearn import plotting

from fmripreprocessing.connectivity.correlation import load_confounds
from fmripreprocessing.utils.data import *
from fmripreprocessing.utils.masks import intersect_multilabel
from fmripreprocessing.utils.visualization import *

sys.path.append("../")
##########################################################################
events = get_event_file()
fmri = get_subject_NF_run(
    path_to_data="/users2/local/alix/out",
    space="MNI152NLin2009cAsym_res-2",
    run_ids=[1],
)
confounds, regressor = load_confounds(events=events, fmri_path=fmri[0])

# %#

# plotting.plot_design_matrix(confounds)

start_finish = []
id = False
for i, t in enumerate(regressor >= 0.5):
    if t and id == False:
        id = True
        start = i
    elif t == False and id:
        id = False
        finish = i
        start_finish.append([start, finish])
    else:
        continue


plt.plot(regressor)
# for i in start_finish:
#    plt.axvspan(i[0], i[1], color= 'green', alpha=0.2)
plt.xlabel("time (s)")
##########################################################################

scheafer = "/homes/a19lamou/fmri_data_proc/data/masks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz"
labels = np.array(
    np.unique(
        intersect_multilabel(
            "/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz",
            scheafer,
        ).get_fdata()
    ),
    dtype=int,
)

for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    regression = "taskReg" if config["task"] else "NOtaskReg"
    ts = "task_block" if config["only_task"] else "all"
    measure = "cor" if config["kind"] == "correlation" else "pcorr"
    global_signal = "basic" if atlas_name == "Sch" else None

    cor = np.load(
        f"/homes/a19lamou/fmri_data_proc/data/connectivity/{measure}_{atlas_name}_run_{1}_{regression}_{ts}.npy"
    )
    if atlas_name == "HMA":
        plot_HMAT_cor_mat(
            np.mean(cor, axis=0),
            vmin=None,
            vmax=None,
            title=f"{config['kind']}-{regression}-{ts}",
        )
    else:
        plot_glasser_cor_mat(
            np.mean(cor, axis=0),
            labels[1:] - 1,
            vmin=None,
            vmax=None,
            title=f"{config['kind']}-{regression}-{ts}-GSR",
        )

for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    regression = "taskReg" if config["task"] else "NOtaskReg"
    ts = "task_block" if config["only_task"] else "all"
    measure = "cor" if config["kind"] == "correlation" else "pcorr"
    global_signal = None if atlas_name == "Sch" else "basic"

    cor = np.load(
        f"/homes/a19lamou/fmri_data_proc/data/connectivity/{measure}_{atlas_name}_run_{1}_{regression}_{ts}_{global_signal}.npy"
    )
    if atlas_name == "HMA":
        plot_HMAT_cor_mat(
            np.mean(cor, axis=0),
            vmin=None,
            vmax=None,
            title=f"{config['kind']}-{regression}-{ts}-GSR",
        )
    else:
        plot_glasser_cor_mat(
            np.mean(cor, axis=0),
            labels[1:] - 1,
            vmin=None,
            vmax=None,
            title=f"{config['kind']}-{regression}-{ts}",
        )


cor = np.load(
    f"/homes/a19lamou/fmri_data_proc/data/connectivity/cor_Sch_run_{1}_NOtaskReg_all.npy"
)
cor2 = np.load(
    f"/homes/a19lamou/fmri_data_proc/data/connectivity/pcorr_Sch_run_{1}_taskReg_all.npy"
)

cor3 = np.load(
    f"/homes/a19lamou/fmri_data_proc/data/connectivity/cor_Sch_run_{1}_taskReg_task_block.npy"
)
cor4 = np.load(
    f"/homes/a19lamou/fmri_data_proc/data/connectivity/cor_Sch_run_{1}_NOtaskReg_task_block.npy"
)

plot_glasser_cor_mat(
    np.mean(cor4, axis=0), labels=labels[1:] - 1, vmin=0.2, vmax=-0.2
)
plot_glasser_cor_mat(
    np.mean(cor4, axis=0), labels=labels[1:] - 1, vmin=0.2, vmax=-0.2
)

plot_glasser_cor_mat(
    np.mean(cor - cor4, axis=0), labels=labels[1:] - 1, vmin=0.2, vmax=-0.2
)
plot_glasser_cor_mat(
    np.mean(cor2 - cor3, axis=0), labels=labels[1:] - 1, vmin=0.2, vmax=-0.2
)


plot_HMAT_cor_mat(np.mean(cor2 - cor3, axis=0), vmin=None, vmax=None)

print(np.min(np.mean(cor - cor2, axis=0)), np.max(np.mean(cor - cor2, axis=0)))
plot_HMAT_cor_mat(np.mean(cor2, axis=0), vmax=0.6, vmin=0)

ppi = np.load(
    "/homes/a19lamou/fmri_data_proc/data/connectivity/PPI_HMA_run_1.npy"
)
plot_HMAT_cor_mat(np.mean(ppi, axis=0), vmin=None, vmax=None)
