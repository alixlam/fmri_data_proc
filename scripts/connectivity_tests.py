import os
import sys
sys.path.append("../")

import warnings
import argparse

import numpy as np

from fmripreprocessing.configs.config_combined import CONFIG
from fmripreprocessing.connectivity.correlation import (
    get_task_FC,
)

warnings.filterwarnings("ignore")

BASEDIR = "/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined/XP2"

parser = argparse.ArgumentParser(
    description='Compute connectivity matrices for selected subjects and config'
)

parser.add_argument('-out', '--o', default=BASEDIR)


for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    strategy = config["denoise_strategy"]
    regression = "taskReg" if config["task"] else "NOtaskReg"
    ts = "task_block" if config["task_block"] else "all"
    measure = "cor" if config["kind"] == "correlation" else "pcorr"
    global_signal = 'basic' if atlas_name == "Sch" else None
    reg_type = "hrf" if config["reg_type"] != "FIR" else "FIR"
    run_ids = (
        config["run_ids"]
        if config["separate_runs"]
        else ["".join(str(i) for i in config["run_ids"])]
    )
    ses_ids = config["ses_id"]
    print("#" * 20)
    print(
        f"Running config: \nAtlas : {atlas_name}\nregression : {regression}\nts : {ts}\nmeasure : {measure}"
    )
    results = get_task_FC(gs=global_signal, **config)
    for ses in ses_ids:
        for run in run_ids:
            filename =f"conn_ses-{ses}_run-{run}_measure-{measure}_ts-{ts}_space-{atlas_name}_densoise-{strategy}_{global_signal}.npy"
            if regression == "taskReg":
                final_path = os.path.join(BASEDIR, regression, reg_type, filename)
            else :
                final_path = os.path.join(BASEDIR, regression, filename)
            if os.path.isdir(os.path.dirname(final_path)) == False:
                os.makedirs(os.path.dirname(final_path))
            mat = np.array([corr[str(ses)][run] for _, corr in results.items()])
            np.save(final_path, mat)
