import os
import sys

sys.path.append("../")


import numpy as np
import scipy.stats as stats
import tqdm
from statsmodels.api import OLS
from statsmodels.stats.multitest import fdrcorrection


def make_sym_matrix(n, vals):
    m = np.zeros([n, n], dtype=np.double)
    xs, ys = np.tril_indices(n, k=-1)
    m[xs, ys] = vals
    m[ys, xs] = vals
    m[np.diag_indices(n)] = 0
    return m

def apply_ttest_correlation(X, X2=None, correction = 'FDR', alpha=0.05, return_uncorrected=False):
    cor_tvals = np.zeros(X.shape[1:3])
    cor_pvals = np.zeros(X.shape[1:3])
    pval_count = 0
    pval_vector = np.ones(int(np.sum(np.tri(X.shape[1], X.shape[2], -1))))
    for row in tqdm.tqdm(range(X.shape[1])):
        for col in range(X.shape[2]):
            if row > col:
                if X2 is not None:
                    result = stats.ttest_rel(
                            np.arctanh(X)[:, row, col],
                            np.arctanh(X2)[:, row, col],
                        )
                else:
                    result = stats.ttest_1samp(np.arctanh(X)[:, row,col], popmean = 0)
                cor_tvals[row,col] = result.statistic
                cor_pvals[row, col] = result.pvalue
                cor_tvals[col,row] = result.statistic
                cor_pvals[col, row] = result.pvalue
                pval_vector[pval_count] = result.pvalue
                pval_count += 1


    fdr_corrected = fdrcorrection(pval_vector, alpha=alpha)[0]
    fdr_corrected = make_sym_matrix(X.shape[1], fdr_corrected)

    if X2 is not None:
        corrected_vals = np.tanh(
                np.mean(
                    np.arctanh(X) - np.arctanh(X2), axis=0
                )
            )
    else:
        corrected_vals = np.tanh(np.mean(np.arctanh(X), axis = 0))
    if return_uncorrected:
        return corrected_vals, fdr_corrected
    else:
        corrected_vals[fdr_corrected == 0] = 0
        return corrected_vals 
    
