import nibabel as nib
import numpy as np
from nilearn.image import binarize_img, resample_to_img


def combine_mask_files(mask_files, mask_method=None, mask_index=None):
    """Combines input mask files into a single nibabel image
    Parameters
    ----------
    mask_files: a list
        one or more binary mask files
    mask_method: enum ('union', 'intersect', 'none')
        determines how to combine masks
    mask_index: an integer
        determines which file to return (mutually exclusive with mask_method)
    Returns
    -------
    masks: a list of nibabel images
    """

    if mask_index or not mask_method:
        if not mask_index:
            if len(mask_files) == 1:
                mask_index = 0
            else:
                raise ValueError(
                    (
                        "When more than one mask file is provided, "
                        "one of merge_method or mask_index must be "
                        "set"
                    )
                )
        if mask_index < len(mask_files):
            mask = nib.load(mask_files[mask_index])
            return [mask]
        raise ValueError(
            (
                "mask_index {0} must be less than number of mask " "files {1}"
            ).format(mask_index, len(mask_files))
        )
    masks = []
    if mask_method == "none":
        for filename in mask_files:
            masks.append(nib.load(filename))
        return masks

    if mask_method == "union":
        mask = None
        for filename in mask_files:
            img = nib.load(filename)
            img_as_mask = np.asanyarray(img.dataobj).astype("int32") > 0
            if mask is None:
                mask = img_as_mask
            np.logical_or(mask, img_as_mask, mask)
        img = nib.Nifti1Image(mask, img.affine, header=img.header)
        return [img]

    if mask_method == "intersect":
        mask = None
        for filename in mask_files:
            img = nib.load(filename)
            img_as_mask = np.asanyarray(img.dataobj).astype("int32") > 0
            if mask is None:
                mask = img_as_mask
            np.logical_and(mask, img_as_mask, mask)
        img = nib.Nifti1Image(mask, img.affine, header=img.header)
        return [img]


def resample_mask_to_bold(bold_img, mask, threshold=0.5, interpolation="nearest"):
    resampled_mask = resample_to_img(mask, bold_img, interpolation= "nearest")
    return binarize_img(resampled_mask, threshold=threshold)
