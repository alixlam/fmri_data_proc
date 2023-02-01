import glob
import os

from nilearn.masking import compute_epi_mask, intersect_masks

from fmripreprocessing.utils.masks import resample_mask_to_bold

from nilearn.datasets import load_mni152_gm_mask


def get_subject_NF_run(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub-xp201",
    run_ids=[1, 2, 3],
    space="MNI152NLin2009cAsym",
):
    files = []
    for id in run_ids:
        files += glob.glob(
            os.path.join(
                path_to_data,
                subject_id,
                "func",
                f"{subject_id}_task-*dNF_run-{id}_space-{space}_desc-preproc_bold.nii.gz",
            )
        )
    return files


def get_subject_brain_mask_from_T1(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub_xp201",
    space="MNI152NLin2009cAsym",
):
    T1_brain_mask = os.path.join(
        path_to_data,
        subject_id,
        "anat",
        f"{subject_id}_space-{space}_desc-brain_mask.nii.gz",
    )
    bold_example = os.path.join(
        path_to_data,
        subject_id,
        "func",
        f"{subject_id}_task-MIpre_space-{space}_desc-preproc_bold.nii.gz",
    )
    mask_fov = compute_epi_mask(bold_example)
    T1_bold = resample_mask_to_bold(bold_example, T1_brain_mask)
    GM_mask_bold = resample_mask_to_bold(bold_example, load_mni152_gm_mask())

    return intersect_masks([T1_bold, mask_fov, GM_mask_bold], 1)


def get_subject_MI(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub-xp201",
    run_ids="pre",
    space="MNI152NLin2009cAsym",
):
    return [
        os.path.join(
            path_to_data,
            subject_id,
            "func",
            f"{subject_id}_task-MI{run_ids}_space-{space}_desc-preproc_bold.nii.gz",
        )
    ]


def get_subjects_functional_data(
    path_to_data="/users2/local/alix/out2",
    subject_id="sub-xp201",
    space="MNI152NLin2009cAsym",
    task="NF",
    run_ids=[1, 2, 3],
):
    assert task in [
        "1dNF",
        "2dNF",
        "MIpre",
        "MIpost",
        "NF",
        "MI",
    ], f"task must be either 1dNF, 2dNF, MIpre or MIpost, instead got {task}"
    if "NF" in task:
        return get_subject_NF_run(path_to_data, subject_id, run_ids, space)
    else:
        return get_subject_MI(path_to_data, subject_id, run_ids, space)


def get_event_file(path_to_data="/users2/local/alix/XP2", task_name="1dNF"):
    return os.path.join(path_to_data, f"task-{task_name}_events.tsv")
