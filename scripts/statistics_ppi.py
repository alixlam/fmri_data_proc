import os
import sys

import numpy as np
import scipy.stats as stats
import tqdm
from statsmodels.api import OLS
from statsmodels.stats.multitest import fdrcorrection

from fmripreprocessing.configs.config_ppi import CONFIG
from fmripreprocessing.utils.masks import intersect_multilabel
from fmripreprocessing.utils.visualization import *

sys.path.append("../")

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


for config in CONFIG:
    atlas_name = os.path.basename(config["atlas"])[:3]
    global_signal = "basic" if atlas_name == "Sch" else None

    print("#" * 20)
    print(f"Running config: \nAtlas : {atlas_name}")

    cor = np.load(
        f"/homes/a19lamou/fmri_data_proc/data/connectivity/unthresh_combined/PPI_{atlas_name}_run_{123}.npy"
    )
    cor_tvals = np.zeros(cor.shape[1:3])
    cor_pvals = np.zeros(cor.shape[1:3])
    pval_count = 0
    pval_vector = np.ones(cor.shape[1] * cor.shape[2])
    for row in tqdm.tqdm(range(cor.shape[1])):
        for col in range(cor.shape[2]):
            result = stats.ttest_1samp(cor[:, row, col], popmean=0)
            cor_tvals[row, col] = result.statistic
            cor_pvals[row, col] = result.pvalue
            pval_vector[pval_count] = result.pvalue
            pval_count += 1

    fdr_corrected = fdrcorrection(pval_vector)[0].reshape(cor_tvals.shape)

    corrected_vals = np.ma.masked_array(
        cor_tvals, mask=np.ones(fdr_corrected.shape) - fdr_corrected
    )
    cor_tvals[fdr_corrected == 0] = 0
    np.save(
        f"/homes/a19lamou/fmri_data_proc/data/connectivity/FDR_cor/PPI_{atlas_name}_run_{123}.npy",
        cor_tvals,
    )
