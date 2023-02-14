import os
import numpy as np
import nibabel as nb
from nibabel.processing import conform
from skimage.transform import resize
from tqdm import tqdm
from scipy.ndimage import rotate

def intensity_normalize_apply_brain_mask_resize_npy(
    mri_post_smriprep_dir, 
    output_dir, 
    output_file_prefix,
    intensity_normalization_mode, # zscore, zero_one, full_values
    intensity_normalization_cutoffs = [0.1, 0.1],
    resize_shape = [182, 218, 182],
    subject_limit = None, # In initial runs, smriprep hasn't been completed on all subjects so to not run on everyone
    subject_exclusions = [],
    trim_images = False,
    trim_parameters = [(28, -2), (20,-20), (20,-18)],
    extra_output_str = ''
):
    """
    This function does the following
        1. Intensity normalization on brain data only (using mask)
            a. z-score: calculate z-scores (mean is zero, values will be above and below based on their stddev)
            b. zero_one: remove values above/below cutoffs (percentiles) and then normalize between 0-1
        2. Apply the brain mask to the normalized data
        3. Conform to a shape
        4. Save to npy
    """

    # Get list of subjects that need masks
    subject_list = [x.split('-')[-1] for x in os.listdir(mri_post_smriprep_dir) if '.' not in x and 'logs' not in x and x.split('-')[-1] not in subject_exclusions]
    
    if not subject_limit:
        subject_limit = len(subject_list)
    
    slices = []
    
    for subject in tqdm(sorted(subject_list)[0:subject_limit]):
        # 1. Intensity normalization
        img = nb.load(os.path.join(
            mri_post_smriprep_dir, 
            'sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'.format(subject, subject))
        )
        img_data = img.get_fdata()
        mask = nb.load(os.path.join(
            mri_post_smriprep_dir, 
            'sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(subject, subject))
        )
        mask_data = mask.get_fdata()
        logical_mask = mask_data > 0.
        
        if intensity_normalization_mode == 'zscore':
            mean = img_data[logical_mask].mean()
            std = img_data[logical_mask].std()
            img_data = (img_data - mean) / std
        elif intensity_normalization_mode == 'zero_one':
            flattened = (img_data * mask_data).flatten()
            min_value = np.percentile(flattened, intensity_normalization_cutoffs[0])
            max_value = np.percentile(flattened, 100-intensity_normalization_cutoffs[1])
            img_data = np.where(img_data<=min_value, min_value, img_data)
            img_data = np.where(img_data>=max_value, max_value, img_data)
            img_data = (img_data-img_data.min()) / (img_data.max()-img_data.min())           
        else:
            print('Please select an intensity_normalization_mode of the following: zscore, zero_one')
            raise 
        
        # 2. Apply mask
        masked = img_data * mask_data

        
        # 3. Resize
        
        # Rotate and fix the axes before resizing
        resized_image = rotate(masked, 90, axes=(0,2))
        resized_image = np.swapaxes(resized_image, 1, 2)
        
        # Remove some dead space around the image
        if trim_images:
            # First set to something standard
            resized_image = resize(resized_image, [182, 182, 218], order=1, preserve_range=True)
            resized_image = resized_image[
                trim_parameters[0][0]:trim_parameters[0][1], 
                trim_parameters[1][0]:trim_parameters[1][1], 
                trim_parameters[2][0]:trim_parameters[2][1]]
        
        resized_image = resize(resized_image, resize_shape, order=1, preserve_range=True)

        slices.append(resized_image)

    # 4. Save to npy
    slices = np.array(slices).astype("float32")
    
    if intensity_normalization_mode == 'zscore':
        output_fname = '{}_{}_{}x{}x{}'.format(
            output_file_prefix, 
            intensity_normalization_mode, 
            resize_shape[0], 
            resize_shape[1],
            resize_shape[2]
        )
    elif intensity_normalization_mode == 'zero_one':
        output_fname = '{}_{}_{}_{}_{}x{}x{}'.format(
            output_file_prefix, 
            intensity_normalization_mode,
            intensity_normalization_cutoffs[0],
            intensity_normalization_cutoffs[1],
            resize_shape[0], 
            resize_shape[1],
            resize_shape[2]
        )
        
    if trim_images:
        output_fname = output_fname + '_trimmed'
        
    outout_fname = output_fname + extra_output_str
        
    output_path = os.path.join(output_dir, output_fname)
    
    np.save(output_path, slices)

def intensity_normalize_apply_brain_mask_patch_npy(
    mri_post_smriprep_dir, 
    output_dir, 
    output_file_prefix,
    intensity_normalization_mode, # zscore, zero_one, full_values
    intensity_normalization_cutoffs = [0.1, 0.1],
    resize_shape = [182, 218, 182],
    subject_limit = None, # In initial runs, smriprep hasn't been completed on all subjects so to not run on everyone
    subject_exclusions = [],
    patch_coords = [(95, 140), (46, 91), (75, 125)],
    patch_name = 'hippocampus',
    bilateral_patch = True,
    second_patch_coords = [(95, 140), (91, 136), (75, 125)]
):
    """
    This function does the following
        1. Intensity normalization on brain data only (using mask)
            a. z-score: calculate z-scores (mean is zero, values will be above and below based on their stddev)
            b. zero_one: remove values above/below cutoffs (percentiles) and then normalize between 0-1
        2. Apply the brain mask to the normalized data
        3. Conform to a shape
        4. Save to npy
    """

    # Get list of subjects that need masks
    subject_list = [x.split('-')[-1] for x in os.listdir(mri_post_smriprep_dir) if '.' not in x and 'logs' not in x and x.split('-')[-1] not in subject_exclusions]
    
    if not subject_limit:
        subject_limit = len(subject_list)
    
    slices = []
    
    for subject in tqdm(sorted(subject_list)[0:subject_limit]):
        # 1. Intensity normalization
        img = nb.load(os.path.join(
            mri_post_smriprep_dir, 
            'sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'.format(subject, subject))
        )
        img_data = img.get_fdata()
        mask = nb.load(os.path.join(
            mri_post_smriprep_dir, 
            'sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(subject, subject))
        )
        mask_data = mask.get_fdata()
        logical_mask = mask_data > 0.
        
        if intensity_normalization_mode == 'zscore':
            mean = img_data[logical_mask].mean()
            std = img_data[logical_mask].std()
            img_data = (img_data - mean) / std
        elif intensity_normalization_mode == 'zero_one':
            flattened = (img_data * mask_data).flatten()
            min_value = np.percentile(flattened, intensity_normalization_cutoffs[0])
            max_value = np.percentile(flattened, 100-intensity_normalization_cutoffs[1])
            img_data = np.where(img_data<=min_value, min_value, img_data)
            img_data = np.where(img_data>=max_value, max_value, img_data)
            img_data = (img_data-img_data.min()) / (img_data.max()-img_data.min())           
        else:
            print('Please select an intensity_normalization_mode of the following: zscore, zero_one')
            raise 
        
        # 2. Apply mask
        masked = img_data * mask_data

        
        # 3. Resize
        
        # Rotate and fix the axes before resizing
        resized_image = rotate(masked, 90, axes=(0,2))
        resized_image = np.swapaxes(resized_image, 1, 2)
        resized_image = resize(resized_image, [182, 182, 218], order=1, preserve_range=True)
        
        if bilateral_patch:
        
            # We can get two image patches
            image_patch_1 = resized_image[
                patch_coords[0][0]:patch_coords[0][1], 
                patch_coords[1][0]:patch_coords[1][1], 
                patch_coords[2][0]:patch_coords[2][1]]
            
            image_patch_2 = resized_image[
                second_patch_coords[0][0]:second_patch_coords[0][1], 
                second_patch_coords[1][0]:second_patch_coords[1][1], 
                second_patch_coords[2][0]:second_patch_coords[2][1]]
            
            # Need to flip image so that it's the same orientation as the other
            image_patch_2 = np.flip(image_patch_2, axis=1)
            
            slices.append(image_patch_1)
            slices.append(image_patch_2)
            
        else:
            
            image_patch = resized_image[
                patch_coords[0][0]:patch_coords[0][1], 
                patch_coords[1][0]:patch_coords[1][1], 
                patch_coords[2][0]:patch_coords[2][1]]

            slices.append(image_patch)

    # 4. Save to npy
    slices = np.array(slices).astype("float32")
    
    if intensity_normalization_mode == 'zscore':
        output_fname = '{}_{}_{}x{}x{}'.format(
            output_file_prefix, 
            intensity_normalization_mode, 
            resize_shape[0], 
            resize_shape[1],
            resize_shape[2]
        )
    elif intensity_normalization_mode == 'zero_one':
        output_fname = '{}_{}_{}_{}_{}x{}x{}'.format(
            output_file_prefix, 
            intensity_normalization_mode,
            intensity_normalization_cutoffs[0],
            intensity_normalization_cutoffs[1],
            resize_shape[0], 
            resize_shape[1],
            resize_shape[2]
        )
        
    output_fname = output_fname + '{}_patch_{}_{}_{}_{}_{}_{}'.format(
        patch_name,
        patch_coords[0][0], patch_coords[0][1], 
        patch_coords[1][0], patch_coords[1][1], 
        patch_coords[2][0], patch_coords[2][1]
    )
        
    output_path = os.path.join(output_dir, output_fname)
    
    np.save(output_path, slices)
    
    
def intensity_normalize_apply_brain_mask_conform(
    mri_post_smriprep_dir, 
    output_dir, 
    conform_shape = [182, 218, 182],
    subject_limit = None, # In initial runs, smriprep hasn't been completed on all subjects so to not run on everyone
    subject_exclusions = []
):
    """
    This function does the following
        1. Intensity normalization using z-scores and only on brain data (using mask)
        2. Apply the brain mask to the normalized data
        3. Conform to a shape
    """
    
    # First, let's see if any images are already in output dir since we can skip those
    be_completed = [x.split('_')[0] for x in os.listdir(output_dir) if 'nii' in x.lower()]

    # Get list of subjects that need masks
    subject_list = [x.split('-')[-1] for x in os.listdir(mri_post_smriprep_dir) if '.' not in x and 'logs' not in x and x not in subject_exclusions]
    subject_list = [x for x in subject_list if x not in be_completed]
    
    if not subject_limit:
        subject_limit = len(subject_list)

    for subject in tqdm(sorted(subject_list)[0:subject_limit]):
        # 1. Z-score intensity normalization
        img = nb.load(os.path.join(
            mri_post_smriprep_dir, 
            'sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'.format(subject, subject))
        )
        img_data = img.get_fdata()
        mask = nb.load(os.path.join(
            mri_post_smriprep_dir, 
            'sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'.format(subject, subject))
        )
        mask_data = mask.get_fdata()
        logical_mask = mask_data > 0.
        mean = img_data[logical_mask].mean()
        std = img_data[logical_mask].std()
        intensity_normalized = nb.Nifti1Image((img_data - mean) / std, img.affine, img.header)
        
        # 2. Apply mask
        masked = intensity_normalized.__class__(
            intensity_normalized.dataobj * mask_data, 
            intensity_normalized.affine, 
            intensity_normalized.header
        )
        
        # 3. Conform
        conformed_img = conform(masked, out_shape = conform_shape)
        conformed_img.to_filename(os.path.join(output_dir, '{}_IN_BE_conformed.nii'.format(subject)))
            
def conform_prob_masks(
    mri_post_smriprep_dir, 
    output_dir, 
    target,
    conform_shape = [182, 218, 182],
    subject_limit = None, # In initial runs, smriprep hasn't been completed on all subjects so to not run on everyone
    subject_exclusions = []
):
    # First, let's see if any images are already in output dir since we can skip those
    be_completed = [x.split('_')[0] for x in os.listdir(output_dir) if target in x]

    # Get list of subjects that need masks
    subject_list = [x.split('-')[-1] for x in os.listdir(mri_post_smriprep_dir) if '.' not in x and 'logs' not in x and x not in subject_exclusions]
    subject_list = [x for x in subject_list if x not in be_completed]
    
    if not subject_limit:
        subject_limit = len(subject_list)

    for subject in tqdm(sorted(subject_list)[0:subject_limit]):
        img = nb.load(os.path.join(
            mri_post_smriprep_dir, 
            'sub-{}/anat/sub-{}_space-MNI152NLin2009cAsym_label-{}_probseg.nii.gz'.format(subject, subject, target))
        )
        
        conformed_img = conform(img, out_shape = conform_shape)
        conformed_img.to_filename(os.path.join(output_dir, '{}_{}_conformed.nii'.format(subject, target)))
        
    
def apply_image_mask(img_file, mask_file):
    img = nb.load(img_file)
    msknii = nb.load(mask_file)
    msk = msknii.get_fdata()
    masked = img.__class__(img.dataobj * msk, None, img.header)
    
    return masked
    
if __name__ == "__main__":
    intensity_normalize_apply_brain_mask_resize_npy(
        'data/matched_images/A_train/mri_post_smriprep/smriprep',
        'data/matched_images/A_train/cls_input',
        output_file_prefix = 'A_train_classifier_input',
        intensity_normalization_mode = 'zero_one',
        resize_shape = [182, 182, 218],
        subject_exclusions = ['0000','0988', '1274'],
        trim_images=True
    )
    
    intensity_normalize_apply_brain_mask_resize_npy(
        'data/matched_images/A_val/mri_post_smriprep/smriprep',
        'data/matched_images/A_val/cls_input',
        output_file_prefix = 'A_val_classifier_input',
        intensity_normalization_mode = 'zero_one',
        resize_shape = [182, 182, 218],
        trim_images=True
    )