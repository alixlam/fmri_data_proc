import  ants
import nibabel as nib
import os, glob

def apply_transform_ants(fixed_image_T1, fixed_image_MNI, moving, transforms):
    fixed_T1 = ants.image_read(fixed_image_T1)
    fixed_MNI = ants.image_read(fixed_image_MNI)
    moving = ants.image_read(moving)
    transformed_img_T1 = ants.apply_transforms(fixed=fixed_T1, moving=moving, transformlist=transforms[:1], interpolator='nearestNeighbor')
    transformed_img_MNI = ants.apply_transforms(fixed=fixed_MNI, moving=moving, transformlist=transforms, interpolator='nearestNeighbor')
    
    return {"T1w": transformed_img_T1, "MNI": transformed_img_MNI}

def get_files(subject_id = "sub-015", path_data = "/users2/local/alix/NF_AVC_tests", pathdata_derivatives = "/users2/local/alix/NF_AVC_derivatives_masks"):
    transforms = [os.path.join(pathdata_derivatives, subject_id, "anat", f"{subject_id}_from-orig_to-T1w_mode-image_xfm.txt"),
                    os.path.join(pathdata_derivatives, subject_id, "anat", f"{subject_id}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5")]

    lesion_mask = os.path.join(path_data, subject_id, "anat", f"{subject_id}_lesion_roi.nii.gz")
    T1w_MNI = os.path.join(pathdata_derivatives, subject_id, "anat", f"{subject_id}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
    T1w = os.path.join(pathdata_derivatives, subject_id, "anat", f"{subject_id}_desc-preproc_T1w.nii.gz")
    return {"transforms": transforms, "lesion": lesion_mask, "MNI": T1w_MNI, "T1": T1w}

def save_lesion_masks_derivatives(subject_id, pathdata, pathdata_derivatives, pathout = "/users/local/alix"):
    files = get_files(subject_id=subject_id, path_data=pathdata, pathdata_derivatives=pathdata_derivatives)
    transformed = apply_transform_ants(files["T1"], files["MNI"], files['lesion'], files['transforms'])
    if pathout is not None:
        if not os.path.isdir(pathout):
            os.makedirs(pathout)
        savepath = pathout
    else: 
        savepath = os.path.join(pathdata_derivatives, subject_id, "anat", f"{subject_id}")
    ants.image_write(transformed["T1w"], os.path.join(savepath, f"{subject_id}_space-T1w_lesion_roi.nii.gz"))
    ants.image_write(transformed["MNI"], os.path.join(savepath, f"{subject_id}_space-MNI152NLin2009cAsym_lesion_roi.nii.gz"))
    return transformed