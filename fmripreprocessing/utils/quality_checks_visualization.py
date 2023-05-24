import sys
sys.path.append("../../rapidtide/rapidtide/scripts")
from nilearn import plotting
from fmripreprocessing.utils.data import get_lagmaps_inputs
import os
import glob
import subprocess
from nilearn import image as nim
from nilearn import surface
from nilearn.connectome import ConnectivityMeasure
import nilearn.masking as msk
import nibabel as nib
import numpy as np
from niworkflows.viz.utils import plot_registration, cuts_from_bbox, compose_view, extract_svg
from uuid import uuid4
import tqdm
import ants
from fmripreprocessing.utils.masks import resample_mask_to_bold
import matplotlib.pyplot as plt
from svgutils.transform import fromstring
from fmripreprocessing.utils.reports import HEMO_TEMPLATE
from fmripreprocessing.utils.data import fetch_sub_fsaverage


def get_lagmaps(subjects_id, pathdata_derivatives, space='MNI152NLin2009cAsym', spatial_filt=2, TR = 1, saving_path=None):
    for subject in subjects_id:
        if saving_path == None:
            saving_path = os.path.join(pathdata_derivatives, subject, "lagmaps")
        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)
        path_lesion, path_GM, path_fov, path_to_bold, path_to_confounds = get_lagmaps_inputs(subject, pathdata_derivatives, space)
        # Run rapidtide to get lagmaps
        if not os.path.isfile(os.path.join(saving_path, f"{subject}_desc-maxtime_map.nii.gz")):
            p = subprocess.run(['rapidtide', str(path_to_bold), 
                                os.path.join(saving_path, subject),
                                '--datatstep', str(TR),
                                '--corrmask', path_GM,
                                '--globalmeaninclude', path_GM,
                                '--globalmeanexclude', path_lesion,
                                '--filterband', 'None',
                                '--spatialfilt', str(spatial_filt),
                                '--searchrange', '-10', '10',
                                '--motionfile', path_to_confounds,
                                '--noglm',
                                '--motderiv', '--motpos'])
        # Mask lagmap
        lagmap = os.path.join(saving_path, f"{subject}_desc-maxtime_map.nii.gz")
        
        #smooth lagmap
        mask_fitcorr = 1 - nib.load(os.path.join(saving_path, f"{subject}_desc-corrfit_mask.nii.gz")).get_fdata()
        mask_fitcorr[mask_fitcorr != 1] = 0
        smoothed_lag_map = nim.smooth_img(lagmap, 6)
        masked_map = np.ma.array(smoothed_lag_map.get_fdata(), mask=mask_fitcorr, fill_value=np.nan).filled()
        masked_map_nib = nib.Nifti1Image(masked_map, nib.load(lagmap).affine)
        nib.save(masked_map_nib, os.path.join(saving_path, f"{subject}_desc-maxtime_map_smoothed.nii.gz"))
        
        #save report
        corrmap = os.path.join(os.path.join(pathdata_derivatives, subject, "lagmaps", f"{subject}_desc-maxcorr_map.nii.gz"))
        
        # Mask hemisphere
        lh_mask = resample_mask_to_bold(path_to_bold, f"/homes/a19lamou/fmri_data_proc/data/masks/lh_space-{space}_mask.nii.gz").get_fdata()
        lh_masked = np.ma.array(masked_map, mask=1 - lh_mask, fill_value=np.nan).filled()
        rh_mask = resample_mask_to_bold(path_to_bold, f"/homes/a19lamou/fmri_data_proc/data/masks/rh_space-{space}_mask.nii.gz").get_fdata()
        rh_masked = np.ma.array(masked_map, mask=1 - rh_mask, fill_value=np.nan).filled()
        
        
        mean_lag = round(np.nanmean(np.abs(masked_map)), 2)
        rh_mean_lag = round(np.nanmean(np.abs(rh_masked)), 2)
        lh_mean_lag = round(np.nanmean(np.abs(lh_masked)), 2)

        corrmap_array = nib.load(corrmap).get_fdata()
        mean_corr = round(np.mean(corrmap_array[corrmap_array > 0]), 2)
        prop_excluded = round(1 - (np.sum(nib.load(os.path.join(saving_path, f"{subject}_desc-corrfit_mask.nii.gz")).get_fdata())/np.sum(nib.load(path_GM).get_fdata())), 2)
        summary = HEMO_TEMPLATE.format(mean_lag = mean_lag, mean_maxcorr = mean_corr,
                                        lh_mean_lag = lh_mean_lag,
                                        rh_mean_lag=rh_mean_lag,
                                        prop_voxel = prop_excluded * 100,
                                        num_voxels = np.sum(mask_fitcorr))
        
        with open(os.path.join(os.path.join(pathdata_derivatives, subject, "figures"), f"{subject}_desc-summary_hemo.html"), "w") as outfile:
            outfile.write(summary)
        
        return True
    
def save_registration_plot_lesion(pathdata_derivatives, subjects_ids = None):
    if subjects_ids == None:
        subjects_ids = glob.glob(os.path.join(pathdata_derivatives, "sub-*"))
        subjects_ids = [os.path.basename(sub) for sub in subjects_ids if "html" not in sub]
    for sub in subjects_ids:
        lesion = os.path.join(pathdata_derivatives, sub, "anat", f"{sub}_space-T1w_lesion_roi.nii.gz")
        anat_path = os.path.join(pathdata_derivatives, sub, "anat", f"{sub}_desc-preproc_T1w.nii.gz")
        
        ncuts = 7
        cuts = cuts_from_bbox(nib.load(lesion), cuts=ncuts)
        
        anat_mask = os.path.join(pathdata_derivatives, sub, "anat", f"{sub}_desc-brain_mask.nii.gz")
        fixed_image = nib.load(anat_path)
        fixed_image = msk.unmask(msk.apply_mask(fixed_image, anat_mask),
                anat_mask)
        anat_plot = plot_registration(anat_nii=fixed_image, contour=nib.load(lesion), div_id="fixed-image", estimate_brightness=True, cuts = cuts, label="fixed", )
        ses = glob.glob(os.path.join(pathdata_derivatives, sub, "ses-*"))
        for s in ses:
            runs = glob.glob(os.path.join(s,"func",  "*space-T1w*_boldref.nii.gz"))
            
            for run in runs:
                outpath = os.path.join(pathdata_derivatives,sub, "figures", os.path.basename(run).replace("_space-T1w_boldref.nii.gz", "_desc-flirtnobbrlesion_bold.svg"))
                #if sos.path.isfile(outpath):
                #    pass
                moving_image = nim.resample_to_img(run, fixed_image)
                moving_image = msk.unmask(msk.apply_mask(moving_image, anat_mask), anat_mask)
                bold_plot = plot_registration(anat_nii=moving_image, contour=nib.load(lesion), div_id="moving-image", estimate_brightness=True, cuts = cuts, label="moving")
                outpath = os.path.join(pathdata_derivatives,sub, "figures", os.path.basename(run).replace("_space-T1w_boldref.nii.gz", "_desc-flirtnobbrlesion_bold.svg"))
                compose_view(anat_plot, bold_plot, out_file = outpath)
    return True

def plot_lagmaps(
    anat_nii,
    stat_map_nii,
    div_id,
    plot_params=None,
    order=("z", "x", "y"),
    cuts=None,
    label=None,
    contour=None,
    compress="auto",
    vmax= 10,
):
    """
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """

    plot_params = {} if plot_params is None else plot_params

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    # nilearn 0.10.0 uses Nifti-specific methods
    anat_nii = nib.Nifti1Image.from_image(anat_nii)

    out_files = []
    if contour:
        contour = nib.Nifti1Image.from_image(contour)

    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        plot_params["display_mode"] = mode
        plot_params["cut_coords"] = cuts[mode]
        if i == 0:
            plot_params["title"] = label
        else:
            plot_params["title"] = None

        # Generate nilearn figure
        display = plotting.plot_stat_map(stat_map_nii,bg_img=anat_nii, vmax=vmax, cmap="jet", **plot_params)   
        if contour is not None:
            display.add_contours(contour, colors="black",filled= True, levels=[0.5], linewidths=0.5)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", "%s-%s-%s" % (div_id, mode, uuid4()), 1)
        out_files.append(fromstring(svg))

    return out_files

def save_lagmaps_plots(pathdata_derivatives, subjects_ids=None):
    if subjects_ids == None:
        subjects_ids = glob.glob(os.path.join(pathdata_derivatives, "sub-*"))
        subjects_ids = [os.path.basename(sub) for sub in subjects_ids if "html" not in sub]
    for sub in subjects_ids:
        outpath = os.path.join(pathdata_derivatives,sub, "figures", f"{sub}_desc-lagmaps_T1w.svg")
        #if os.path.isdir(outpath):
        #    pass
        lagmap = os.path.join(pathdata_derivatives, sub, "lagmaps", f"{sub}_desc-maxtime_map_smoothed.nii.gz")
        lesion = os.path.join(pathdata_derivatives, sub, "anat", f"{sub}_space-MNI152NLin2009cAsym_lesion_roi.nii.gz")
        anat_path = os.path.join(pathdata_derivatives, sub, "anat", f"{sub}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz")
        ncuts = 7
        cuts = cuts_from_bbox(nib.load(lesion), cuts=ncuts)
        lag_map_plot = plot_lagmaps(stat_map_nii= lagmap, anat_nii=nib.load(anat_path), contour=nib.load(lesion), div_id="Lagmap", cuts = cuts, order=('z'), label="LagMap")
        outpath = os.path.join(pathdata_derivatives,sub, "figures", f"{sub}_desc-lagmaps_T1w.svg")
        compose_view(lag_map_plot, [], out_file = outpath)
        
        outpath = os.path.join(pathdata_derivatives,sub, "figures", f"{sub}_desc-corrmap_T1w.svg")
        corrmap = os.path.join(os.path.join(pathdata_derivatives, sub, "lagmaps", f"{sub}_desc-maxcorr_map.nii.gz"))
        corr_map_plot = plot_lagmaps(stat_map_nii= corrmap, anat_nii=nib.load(anat_path), contour=nib.load(lesion), div_id="corrmap", cuts = cuts, order=('z'), vmax = 1, label="MaxCorrMap")
        compose_view(corr_map_plot, [], out_file = outpath)
    return True

def get_surface_homotopic(pathdata_derivatives, subjects_ids = None):
    if subjects_ids == None:
        subjects_ids = glob.glob(os.path.join(pathdata_derivatives, "sub-*"))
        subjects_ids = [os.path.basename(sub) for sub in subjects_ids if "html" not in sub]
    for sub in tqdm.tqdm(subjects_ids):
        ses = glob.glob(os.path.join(pathdata_derivatives, sub, "ses-*"))
        for s in ses:
            runs = glob.glob(os.path.join(s,"func",  "*hemi-L_space-fsaverage5_bold.func.gii"))
            for run in runs:
                func_lh = run
                func_rh = run.replace("hemi-L", "hemi-R")
                times_series_lh = surface.load_surf_data(func_lh)
                times_series_rh = surface.load_surf_data(func_rh)
                time_series = np.vstack([times_series_lh, times_series_rh])
                correlation = ConnectivityMeasure(kind='correlation')
                corr_map = correlation.fit_transform([time_series.T])
                hC = np.zeros((times_series_lh.shape[0]))
                for i in range(hC.shape[0]):
                    hC[i] = corr_map[0, i, i + hC.shape[0]]
                outpath_lh = os.path.join(pathdata_derivatives, sub, "homotopic", os.path.basename(run).replace("hemi-L_space-fsaverage5_bold.func.gii", "lh.HC"))
                outpath_rh = os.path.join(pathdata_derivatives, sub, "homotopic", os.path.basename(run).replace("hemi-L_space-fsaverage5_bold.func.gii", "rh.HC"))
                
                if not os.path.isdir(os.path.dirname(outpath_lh)):
                    os.makedirs(os.path.dirname(outpath_lh))
                nib.freesurfer.io.write_morph_data(file_like= outpath_lh, values=hC)
                nib.freesurfer.io.write_morph_data(file_like= outpath_rh, values=hC)
    return 0

def get_volume_homotopic(pathdata_derivatives, subjects_id = None):
    for sub in tqdm.tqdm(subjects_id):
        pial_surf = os.path.join(pathdata_derivatives, "sourcedata", "freesurfer", sub, "surf", "lh.pial")
        ribbon = os.path.join(pathdata_derivatives, "sourcedata", "freesurfer", sub, "mri", "ribbon.mgz")

        ses = glob.glob(os.path.join(pathdata_derivatives, sub, "ses-*"))
        for s in ses:
            runs = glob.glob(os.path.join(s, "homotopic",  "*lh.HC"))
            for run in runs:
                pathout = os.path.join(pathdata_derivatives, sub, "homotopic",  os.path.basename(run).replace("lh.HC", "lh_HC.nii.gz"))
                if os.path.isfile(pathout):
                    pass
                # step 1 : transform in subjects native space
                p1 = subprocess.run(['mri_surf2surf', 
                                '--srcsubject', 'fsaverage5',
                                '--trgsubject', sub,
                                '--hemi', 'lh',
                                '--srcsurfval', run,
                                '--trgsurfval', run.replace("lh.HC", "lh_fsnative.HC"),
                                '--trg_type', 'curv'])
                
                # step 2 : transform to volume
                p2 = subprocess.run(['mri_surf2vol',
                                '--o', pathout,
                                '--so', pial_surf, run.replace("lh.HC", "lh_fsnative.HC"),
                                '--ribbon', ribbon])
                
    return 0

def plot_atlas_on_surface(surface, labels, div_id,mode=None, compress="auto", label=None, pathdata_fsaverage= None):
    out_files = []
    display, axes = plt.subplots(nrows=2, ncols=4, subplot_kw={'projection': '3d'})
    surface_av, labels_av = fetch_sub_fsaverage(pathdata_fsaverage, 'fsaverage5')
    for k, (surf, lab) in enumerate(zip([surface, surface_av], [labels, labels_av])):
        for i, hemi in enumerate(["left", "right"]):
            for j, view in enumerate(["lateral", "medial"]):
                if i == 0 and k == 0 and j == 0:
                    title= "Subject"
                elif i==0 and k ==1 and j == 0 :
                    title= "Template"
                else :
                    title = None
                ax = axes[j,i+ k*(k + 1)]
                ax.set_facecolor('black')
                if title is not None:
                    ax.set_title(title, fontdict={'color': 'white'})
                plotting.plot_surf_roi(
                    surf[f"pial_{hemi}"],
                    roi_map = lab[f"atlas_{hemi}"],
                    bg_map= surf[f"sulc_{hemi}"],
                    hemi = hemi,
                    view= view,
                    axes= ax,
                    figure=display,
                    title=title,
                    bg_on_data=True,
                    engine='matplotlib',
                    )
        display.set_facecolor('black')
        svg = extract_svg(display, compress=compress, nilearn=False)
        
        plt.close(display)

        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", "%s-%s-%s" % (div_id, mode, uuid4()), 1)
        out_files.append(fromstring(svg))
    return out_files

def save_surface_plots(pathdata_derivatives, subjects_ids=None):
    if subjects_ids == None:
        subjects_ids = glob.glob(os.path.join(pathdata_derivatives, "sub-*"))
        subjects_ids = [os.path.basename(sub) for sub in subjects_ids if "html" not in sub]
    
    for sub in subjects_ids:
        outpath = os.path.join(pathdata_derivatives,sub, "figures", f"{sub}_desc-surfaceatlas_T1w.svg")
        #if os.path.isdir(outpath):
        #    pass
        surface_sub, atlas_sub = fetch_sub_fsaverage(pathdata_derivatives, sub)
        surface_sub_plot = plot_atlas_on_surface(surface_sub, atlas_sub, div_id="surface", label="Parcellation", pathdata_fsaverage=pathdata_derivatives)
        #surface_sub_plot.savefig(outpath)
        compose_view(surface_sub_plot, [], out_file = outpath)

    return True