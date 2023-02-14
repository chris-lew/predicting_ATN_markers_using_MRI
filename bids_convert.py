import os
from os.path import join
from tqdm import tqdm

from nibabel import load
from nibabel.funcs import squeeze_image


def bids_convert(input_dir, bids_dir):
    # Converts a dir of nii files into a BIDS format
    #
    # Additionally, our images have an extra dimension, this squeezes it down to 3d. 
    #
    
    bids_dir_files = os.listdir(bids_dir)
    
    input_dir_files = os.listdir(input_dir)
    
    image_list = [x for x in input_dir_files if 'sub-{}'.format(x.split('.')[0]) not in bids_dir_files]

    for image in tqdm(sorted(image_list)):
        
        if image.split('.')[1] == 'nii':
        
            subject_name = image.split('.')[0]
            
            subject_dir = join(bids_dir,'sub-{}'.format(subject_name))
            os.mkdir(subject_dir)
            
            subject_dir = join(subject_dir, 'anat')
            os.mkdir(subject_dir)

            try:
                image_data = load(join(input_dir, image))
                squeezed_image = squeeze_image(image_data)
                filename = 'sub-{}_T1w.nii'.format(subject_name)
                squeezed_image.to_filename(join(subject_dir, filename))
            except:
                print(subject_name, 'failure due to img data issue')

if __name__ == "__main__":
    bids_convert('data/matched_images/A_train/mri_label',
                 'data/matched_images/A_train/mri_bids')