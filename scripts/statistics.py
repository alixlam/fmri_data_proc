import os
import sys

sys.path.append("../")


import numpy as np
from fmripreprocessing.connectivity.statistics import apply_ttest_correlation

from fmripreprocessing.configs.config_combined import CONFIG
import argparse


# scheafer = '/homes/a19lamou/fmri_data_proc/data/masks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
# labels = np.array(np.unique(intersect_multilabel("/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz", scheafer).get_fdata()), dtype=int)

BASEDIR = "/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined/XP2"
BASEDIROUT = "/homes/a19lamou/fmri_data_proc/data/connectivity/FDR_cor_combined/XP2"

parser = argparse.ArgumentParser(
    description='Compute connectivity matrices for selected subjects and config'
)

parser.add_argument('-i', '--indir', default=BASEDIR)

parser.add_argument('-o', '--out', default=BASEDIROUT)

parser.add_argument('-s', '--ses', default="None")

parser.add_argument('-r', '--run', default='123')

parser.add_argument('-ti', '--test_id', default="default", choices=["default", "taskReg"])
args = parser.parse_args()


if args.test_id == "default":
    for config in CONFIG:
        atlas_name = os.path.basename(config["atlas"])[:3]
        regression = "taskReg" if config["task"]==True else "NOtaskReg"
        strategy = config["denoise_strategy"]
        ts = "task_block" if config["task_block"]==True else "all"
        measure = "cor" if config["kind"] == "correlation" else "pcorr"
        global_signal = None if atlas_name == "Sch" else None
        reg_type = "hrf" if config["reg_type"] != "FIR" else "FIR"


        print(
            "#"*20
        )
        print(f"Running config: \nAtlas : {atlas_name}\nregression : {regression}\nts : {ts}\nmeasure : {measure}")
        filename =f"conn_ses-{args.ses}_run-{args.run}_measure-{measure}_ts-{ts}_space-{atlas_name}_densoise-{strategy}_{global_signal}.npy"
        if regression == "taskReg":
            final_path_in = os.path.join(args.indir, regression, reg_type, filename)
        else :
            final_path_in = os.path.join(args.indir, regression, filename)
        cor = np.load(final_path_in)
        corrected_vals = apply_ttest_correlation(X=cor, correction="FDR", alpha=0.05, return_uncorrected=False)
        if regression == "taskReg":
            final_path_out = os.path.join(args.out, regression, reg_type, filename)
        else :
            final_path_out = os.path.join(args.out, regression, filename)
        
        if os.path.isdir(os.path.dirname(final_path_out)) == False:
                os.makedirs(os.path.dirname(final_path_out))
        np.save(final_path_out, corrected_vals)

elif args.test_id == "taskReg":
    ATLAS = ["Sch"]
    CON_MEASURE = ["cor"]
    TS = ["task_block"]
    TASKREG = ["FIR"]
    STRAT = "None"
    GS = "basic"

    for atlas_name in ATLAS:
        for measure in CON_MEASURE:
            for ts in TS:
                for taskreg in TASKREG:
                    print("#" * 20)
                    print(
                        f"Running config: \nAtlas : {atlas_name}\nreg : {ts}\nmeasure : {measure}"
                    )
                    filename = f"conn_ses-{args.ses}_run-{args.run}_measure-{measure}_ts-{ts}_space-{atlas_name}_densoise-{STRAT}_{GS}.npy"
                    TaskReg_filename = os.path.join(args.indir, "taskReg", taskreg, filename)
                    NOTaskReg_filename = os.path.join(args.indir, "NOtaskReg", filename)
                    corTaskReg = np.load(TaskReg_filename)
                    corNOTaskReg = np.load(NOTaskReg_filename)
                    corrected_vals = apply_ttest_correlation(X=corNOTaskReg, X2=corTaskReg, alpha=0.05)
                    outpath = os.path.join(args.out, "NOtaskReg-TaskReg", filename)
                    if os.path.isdir(os.path.dirname(outpath)) == False:
                        os.makedirs(os.path.dirname(outpath))
                    np.save(
                        outpath,
                        corrected_vals,)


"""for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    regression = "taskReg" if config["task"]==True else "NOtaskReg"
    ts = "task_block" if config["only_task"]==True else "all"
    measure = "cor" if config["kind"] == "correlation" else "pcorr"
    global_signal = 'basic' if atlas_name == "Sch" else None

    print(
    "#"*20
    )
    print(f"Running config: \nAtlas : {atlas_name}\nregression : {regression}\nts : {ts}\nmeasure : {measure}")

    corRun1 = np.load(f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh/{measure}_{atlas_name}_run_{1}_{regression}_{ts}_{global_signal}.npy")
    corRun3 =  np.load(f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh/{measure}_{atlas_name}_run_{3}_{regression}_{ts}_{global_signal}.npy")
    cor_tvals = np.zeros(corRun1.shape[1:3])
    cor_pvals = np.zeros(corRun1.shape[1:3])
    pval_count = 0
    pval_vector = np.ones(int(np.sum(np.tri(corRun1.shape[1], corRun1.shape[2], -1))))
    for row in tqdm.tqdm(range(corRun1.shape[1])):
        for col in range(corRun1.shape[2]):
            if row > col:
                result = stats.ttest_rel(np.arctanh(corRun3)[:, row,col], np.arctanh(corRun1)[:, row, col])
                cor_tvals[row,col] = result.statistic
                cor_pvals[row, col] = result.pvalue
                cor_tvals[col,row] = result.statistic
                cor_pvals[col, row] = result.pvalue
                pval_vector[pval_count] = result.pvalue
                pval_count += 1


    fdr_corrected = fdrcorrection(pval_vector)[0]
    fdr_corrected = make_sym_matrix(corRun1.shape[1], fdr_corrected)

    corrected_vals = np.ma.masked_array(np.tanh(np.mean(np.arctanh(corRun3) - np.arctanh(corRun1), axis = 0)), mask = np.ones(fdr_corrected.shape) - fdr_corrected)
    test = np.tanh(np.mean(np.arctanh(corRun3) - np.arctanh(corRun1), axis = 0))
    test[fdr_corrected == 0] = 0
    np.save(f"/homes/a19lamou/fmri_data_proc/data/connectivity/FDR_cor_runs/run3-run1_{measure}_{atlas_name}_{regression}_{ts}_{global_signal}.npy", test)
"""
