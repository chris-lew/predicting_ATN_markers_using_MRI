import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def display_image(img, low=0, high=200):
    
    '''
    img_list: a nii file
    low: index of the first slice to be shown.
    high: index of the last slice to be shown.
    '''
    
    # Load str
    img_nib = nb.load(img)
    can = nb.as_closest_canonical(img_nib)
    # Make sure correct dimensions
    if len(img_nib.header.get_data_shape())==4:
        arr = can.get_fdata()[:,:,:,0]
    # If we don't have 4 dim, then need to probably unpack
    else:
        arr = can.get_fdata()

    n_col = 5
    n_row = int((high-low+1)/n_col)
    plt.figure(figsize=(n_col*3, n_row*3))
    for row in range(n_row):
        for col in range(n_col):
            i = n_col*row + col + 1  # index of the subplot
            plt.subplot(n_row, n_col, i)
            plt.axis('off')
            plt.imshow(arr[:,i+low-1,:].T, cmap='gray', origin='lower')
            plt.title('slice-' + str(i+low-1))
    plt.show()

def display_image_slices(img, slices=[106, 110, 114], zscored=False):
    # Load str
    img_nib = nb.load(img)
    can = nb.as_closest_canonical(img_nib)
    # Make sure correct dimensions
    if len(img_nib.header.get_data_shape())==4:
        arr = can.get_fdata()[:,:,:,0]
    # If we don't have 4 dim, then need to probably unpack
    else:
        arr = can.get_fdata()
    n_col = len(slices)
    plt.figure(figsize=(n_col*6, 8))
    for col in range(n_col):
        plt.subplot(1, n_col, col+1)
        plt.axis('off')
        if zscored:
            plt.imshow(arr[:,slices[col],:].T, cmap='gray', origin='lower', vmin=-12, vmax=12)
        else:
            plt.imshow(arr[:,slices[col],:].T, cmap='gray', origin='lower')
        plt.title('slice: ' + str(slices[col]))
    plt.show()
    
def display_images(img_list, low=0, high=200):
    '''
    img_list: a list of images which can be nii file or array.
    low: index of the first slice to be shown.
    high: index of the last slice to be shown.
    '''
    # Change to list of images if not already
    if type(img_list) != list:
        img_list = [img_list]
    
    # If list is a size of 1, don't need to iterate
    if len(img_list) == 1:
        # If it's a string, load it 
        if isinstance(img_list[0], str):
            # Load str
            img_nib = nb.load(img_list[0])
            can = nb.as_closest_canonical(img_nib)
            # Make sure correct dimensions
            if len(img_nib.header.get_data_shape())==4:
                arr = can.get_fdata()[:,:,:,0]
            # If we don't have 4 dim, then need to probably unpack
            else:
                arr = can.get_fdata()
        else:
            # Make sure correct dimensions
            if len(img_list[0].shape)==4:
                arr = img_list[0][:,:,:,0]
            else:
                arr = img_list[0]
        n_col = 5
        n_row = int((high-low+1)/n_col)
        plt.figure(figsize=(n_col*3, n_row*3))
        for row in range(n_row):
            for col in range(n_col):
                i = n_col*row + col + 1  # index of the subplot
                plt.subplot(n_row, n_col, i)
                plt.imshow(arr[:,i+low-1,:].T, cmap='gray', origin='lower')
                plt.title('slice-' + str(i+low-1))
        plt.show()
    else:    
        vol_list = []
        for img in img_list:
            if isinstance(img, str):
                img_nib = nb.load(img)
                can = nb.as_closest_canonical(img_nib)
                if len(img_nib.header.get_data_shape())==4:
                    arr = can.get_fdata()[:,:,:,0]
                else:
                    arr = can.get_fdata()
                vol_list.append(arr)
            else:
                if len(img.shape)==4:
                    vol_list.append(img[:,:,:,0])
                else:
                    vol_list.append(img)
            
        n_col = len(vol_list)
        n_row = int(high-low+1)
        plt.figure(figsize=(5*n_col,5*n_row))
        for s in range(n_row):
            for i in range(n_col):
                plt.subplot(n_row, n_col, s*n_col+i+1)
                if s < vol_list[i].shape[1]:
                    plt.imshow(vol_list[i][:,s+low,:].T, cmap='gray', origin='lower')
                    plt.title('vol-{}, slice-{}'.format(i+1, s+low))
        plt.show()
        
def norm(nii, cut1=0.01, cut2=0.01):
    '''
    cut pixel values from two edges in the histogram of the image.
    nii: the input .nii image.
    cut1: the percentage to cut from the lower edge in the histogram.
    cut2: the percentage to cut from the higher edge in the histogram.
    '''
    img = nb.load(nii)
    arr = img.get_fdata()
    n = arr.flatten().shape[0]
    l = list(arr.flatten())
    l.sort()
    a = l[int(cut1*(n-1))]
    b = l[int((1-cut2)*(n-1))]
    arr = np.where(arr<=a, a, arr)
    arr = np.where(arr>=b, b, arr)
    arr = (arr-arr.min()) / (arr.max()-arr.min())
    return arr

def take_slices(folder, suffix, slice_indices=[106,110,114]):
    '''
    This method is used to take slices from every volumes in a folder to create an array
    folder: the folder all images are stored.
    n1: number of the first slice to take from each single image.
    n2: number of the second slice to take from each single image.
    n3: number of the third slice to take from each single image.
    cut1: the percentage to cut from the left-side end in the histogram of the image
    cut2: the percentage to cut from the right-side end in the histogram of the image
    '''
    
    slices=[]
    file_list = [x for x in os.listdir(folder) if x.endswith(suffix)]
    
    for image in tqdm(sorted(file_list)):
        arr = nb.load(os.path.join(folder, image)).get_fdata()
        slice_list = []
        for s in slice_indices:
            slice_list.append(arr[:, s, :])
        slice_array = np.stack(slice_list)

        sample = np.swapaxes(np.swapaxes(slice_array, 0, 1), 1, 2)
        slices.append(np.float32(sample))
    slices = np.array(slices)
    return slices

def take_all_slices(folder, suffix):
    slices=[]
    file_list = [x for x in os.listdir(folder) if x.endswith(suffix)]
    
    for image in tqdm(sorted(file_list)):
        arr = nb.load(os.path.join(folder, image)).get_fdata()
        arr = np.swapaxes(arr, 1, 2)

        slices.append(np.float32(arr))
    slices = np.array(slices)
    return slices

def resample_and_take_all_slices(
    input_dir,
    input_suffix,
    resample_shape = [96, 104, 96],
    subject_limit = None, # In initial runs, smriprep hasn't been completed on all subjects so to not run on everyone
    subject_exclusions = []
):
    from nibabel.processing import conform

    file_list = [x for x in os.listdir(input_dir) if x.endswith(input_suffix)]
    
    slices=[]
    
    if not subject_limit:
        subject_limit = len(file_list)

    for image in tqdm(sorted(file_list)[0:subject_limit]):
        img = nb.load(os.path.join(input_dir, image))
        
        conformed_img = conform(img, out_shape = resample_shape)
        arr = conformed_img.get_fdata()
        arr = np.swapaxes(arr, 1, 2)

        slices.append(np.float32(arr))
    
    slices = np.array(slices)
    return slices

def display_100_subject_images(image_dir, file_list, zscored=False):
    plt.figure(figsize=(15, 60))
    
    for i in range(100):
        img = nb.load(os.path.join(image_dir, file_list[i]))
        img_arr = nb.as_closest_canonical(img).get_fdata()

        plt.subplot(20, 5, i+1)
        plt.axis('off')
        if zscored:
            plt.imshow(img_arr[:, 110, :].T, cmap='gray', origin='lower', vmin=-12, vmax=12)
        else:
            plt.imshow(img_arr[:, 110, :].T, cmap='gray', origin='lower')
        plt.title('subject: ' + file_list[i].split('_')[0])

    plt.show()
    
def display_100_subject_images_npy(npy_path, target_slice = 110, zscored=False):
    from scipy.ndimage import rotate
    
    plt.figure(figsize=(15, 60))
    img_data = np.load(npy_path)
    
    for i in range(100):
        plt.subplot(20, 5, i+1)
        plt.axis('off')
        if zscored:
            plt.imshow(img_data[i , :, target_slice, :], cmap='gray', vmin=-12, vmax=12)
        else:
            plt.imshow(img_data[i , :, target_slice, :], cmap='gray')
        plt.title('subject: {:04d}'.format(i))

    plt.show()