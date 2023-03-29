import os
import sys

sys.path.append("../")


import numpy as np
import scipy.stats as stats
import tqdm
from statsmodels.api import OLS
from statsmodels.stats.multitest import fdrcorrection

from fmripreprocessing.configs.config_combined import CONFIG
from fmripreprocessing.utils.masks import intersect_multilabel
from fmripreprocessing.utils.visualization import *


# scheafer = '/homes/a19lamou/fmri_data_proc/data/masks/Schaefer2018_400Parcels_7Networks_order_FSLMNI152_2mm.nii.gz'
# labels = np.array(np.unique(intersect_multilabel("/homes/a19lamou/fmri_data_proc/data/masks/global_mask.nii.gz", scheafer).get_fdata()), dtype=int)


def apply_ttest(y, X, con):
    t_test = OLS(y, X).fit().t_test(con)
    return float(t_test.tvalue), float(t_test.pvalue)


def fdr(pvals, tvals):
    return fdrcorrection(pvals, alpha=0.05)[0] * tvals


def make_sym_matrix(n, vals):
    m = np.zeros([n, n], dtype=np.double)
    xs, ys = np.tril_indices(n, k=-1)
    m[xs, ys] = vals
    m[ys, xs] = vals
    m[np.diag_indices(n)] = 0
    return m


"""for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    regression = "taskReg" if config["task"]==True else "NOtaskReg"
    strategy = config["regression"]
    ts = "task_block" if config["only_task"]==True else "all"
    measure = "cor" if config["kind"] == "correlation" else "pcorr"
    global_signal = None if atlas_name == "Sch" else None
    reg_type = "hrf" if config["reg_type"] != "FIR" else "FIR"


    print(
        "#"*20
    )
    print(f"Running config: \nAtlas : {atlas_name}\nregression : {regression}\nts : {ts}\nmeasure : {measure}")
    if regression == "NOtaskReg":
        continue
    if atlas_name == "Sch" or atlas_name == "HMA":
        cor = np.load(f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined/{measure}_{atlas_name}_run_{123}_{regression}_{reg_type}_{ts}_{global_signal}_{strategy}.npy")
        #cor = np.concatenate([np.load(f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh/{measure}_{atlas_name}_run_{i}_{regression}_{ts}_{global_signal}.npy") for i in range(1,4)], axis = 0)
        cor_tvals = np.zeros(cor.shape[1:3])
        cor_pvals = np.zeros(cor.shape[1:3])
        pval_count = 0
        pval_vector = np.ones(int(np.sum(np.tri(cor.shape[1], cor.shape[2], -1))))
        for row in tqdm.tqdm(range(cor.shape[1])):
            for col in range(cor.shape[2]):
                if row > col:
                    result = stats.ttest_1samp(np.arctanh(cor)[:, row,col], popmean = 0)
                    cor_tvals[row,col] = result.statistic
                    cor_pvals[row, col] = result.pvalue
                    cor_tvals[col,row] = result.statistic
                    cor_pvals[col, row] = result.pvalue
                    pval_vector[pval_count] = result.pvalue
                    pval_count += 1


        fdr_corrected = fdrcorrection(pval_vector)[0]
        fdr_corrected = make_sym_matrix(cor.shape[1], fdr_corrected)

        corrected_vals = np.ma.masked_array(np.tanh(np.mean(np.arctanh(cor), axis = 0)), mask = np.ones(fdr_corrected.shape) - fdr_corrected)
        test = np.tanh(np.mean(np.arctanh(cor), axis = 0))
        test[fdr_corrected == 0] = 0
        np.save(f"/homes/a19lamou/fmri_data_proc/data/connectivity/FDR_cor/{measure}_{atlas_name}_run_{123}_{regression}_{reg_type}_{ts}_{global_signal}_{strategy}.npy", test)"""


ATLAS = ["Sch"]
CON_MEASURE = ["cor"]
TASK_REG = ["taskReg", "NOtaskReg"]
TS = ["task_block", "all"]

STRAT = "compcor"
GS = "None"

for atlas in ATLAS:
    for measure in CON_MEASURE:
        for ts in TS:
            print("#" * 20)
            print(
                f"Running config: \nAtlas : {atlas}\nreg : {ts}\nmeasure : {measure}"
            )

            corTaskReg = np.load(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined_ext/{measure}_{atlas}_run_{123}_taskReg_FIR_{ts}_{GS}_{STRAT}.npy"
            )
            corNOTaskReg = np.load(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined_ext/{measure}_{atlas}_run_{123}_NOtaskReg_FIR_{ts}_{GS}_{STRAT}.npy"
            )
            cor_tvals = np.zeros(corTaskReg.shape[1:3])
            cor_pvals = np.zeros(corTaskReg.shape[1:3])
            pval_count = 0
            pval_vector = np.ones(
                int(
                    np.sum(
                        np.tri(corTaskReg.shape[1], corTaskReg.shape[2], -1)
                    )
                )
            )
            for row in tqdm.tqdm(range(corTaskReg.shape[1])):
                for col in range(corTaskReg.shape[2]):
                    if row > col:
                        result = stats.ttest_rel(
                            np.arctanh(corNOTaskReg)[:, row, col],
                            np.arctanh(corTaskReg)[:, row, col],
                        )
                        cor_tvals[row, col] = result.statistic
                        cor_pvals[row, col] = result.pvalue
                        cor_tvals[col, row] = result.statistic
                        cor_pvals[col, row] = result.pvalue
                        pval_vector[pval_count] = result.pvalue
                        pval_count += 1

            fdr_corrected = fdrcorrection(pval_vector)[0]
            fdr_corrected = make_sym_matrix(corTaskReg.shape[1], fdr_corrected)

            corrected_vals = np.ma.masked_array(
                np.tanh(
                    np.mean(
                        np.arctanh(corNOTaskReg) - np.arctanh(corTaskReg),
                        axis=0,
                    )
                ),
                mask=np.ones(fdr_corrected.shape) - fdr_corrected,
            )
            test = np.tanh(
                np.mean(
                    np.arctanh(corNOTaskReg) - np.arctanh(corTaskReg), axis=0
                )
            )
            np.save(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/FDR_cor_ext/NOtaskReg-taskReg_FIR_{measure}_{atlas}_run_{123}_{ts}_{STRAT}_{GS}_uncor.npy",
                test,
            )
            test[fdr_corrected == 0] = 0
            np.save(
                f"/homes/a19lamou/fmri_data_proc/data/connectivity/FDR_cor_ext/NOtaskReg-taskReg_FIR_{measure}_{atlas}_run_{123}_{ts}_{STRAT}_{GS}.npy",
                test,
            )


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
