import os
import sys

import numpy as np

from fmripreprocessing.configs.config_ppi import CONFIG
from fmripreprocessing.connectivity.PPI_clean import run_ppi

sys.path.append("../")

for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    run_ids = (
        config["run_ids"]
        if config["separate_runs"]
        else ["".join(str(i) for i in config["run_ids"])]
    )

    print("#" * 20)
    print(f"Running config: \nAtlas : {atlas_name}")
    results = run_ppi(**config)

    for run in run_ids:
        try:
            mat = np.array([corr[run] for _, corr in results.items()])
            np.save(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined/PPI_{atlas_name}_run_{run}.npy",
                mat,
            )

        except Exception as ex:
            print(ex)
            continue
