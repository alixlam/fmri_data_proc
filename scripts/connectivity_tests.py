import os
import sys
sys.path.append("../")

import warnings

import numpy as np

from fmripreprocessing.configs.config_data_ext import CONFIG
from fmripreprocessing.connectivity.correlation import (
    run_task_reg_correlation_single,
)

warnings.filterwarnings("ignore")

for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    strategy = config["regression"]
    regression = "taskReg" if config["task"] else "NOtaskReg"
    ts = "task_block" if config["only_task"] else "all"
    measure = "cor" if config["kind"] == "correlation" else "pcorr"
    global_signal = 'basic' if atlas_name == "Sch" else None
    reg_type = "hrf" if config["reg_type"] != "FIR" else "FIR"
    run_ids = (
        config["run_ids"]
        if config["separate_runs"]
        else ["".join(str(i) for i in config["run_ids"])]
    )

    """if regression == "NOtaskReg":
        continue"""
    print("#" * 20)
    print(
        f"Running config: \nAtlas : {atlas_name}\nregression : {regression}\nts : {ts}\nmeasure : {measure}"
    )
    results = run_task_reg_correlation_single(gs=global_signal, **config)

    for run in run_ids:
        try:
            mat = np.array([corr[run] for _, corr in results.items()])
            np.save(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined_ext/{measure}_{atlas_name}_run_{run}_{regression}_{reg_type}_{ts}_{global_signal}_{strategy}.npy",
                mat,
            )

        except Exception as ex:
            print(ex)
            continue
