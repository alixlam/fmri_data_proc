import pandas as pd
from nilearn import plotting
from nilearn.glm.second_level import SecondLevelModel
from scipy.stats import norm


def run_second_model(data, plot=True, p_val_thresh=0.001):
    design_matrix = pd.DataFrame([1] * len(data), columns="intercept")
    second_level_glm = SecondLevelModel(smoothing_fwhm=4)
    second_level_glm.fit(data, design_matrix=design_matrix)

    contrast = second_level_glm.compute_contrast(
        second_level_contrast="interset",
        output_type="z_score",
    )

    if plot:
        display = plotting.plot_glass_brain(
            contrast,
            threshold=norm.isf(p_val_thresh),
            colorbar=True,
            display_mode="lzry",
            plot_abs=False,
            title="[TaskNF - Rest] Group maps",
        )

    return contrast, second_level_glm
