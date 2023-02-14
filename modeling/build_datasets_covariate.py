import numpy as np
import tensorflow as tf
import keras
import os
import pandas as pd
from pathlib import Path
from scipy import ndimage
import random

column_label_dict = {
    'A_train': 'A_MEDIAN_CLS',
    'A_val': 'A_MEDIAN_CLS',
    'A_test': 'A_MEDIAN_CLS',
    'T_train': 'tau_CLS', 
    'T_val': 'tau_CLS', 
    'T_test': 'tau_CLS',
    'N_train': 'N_CLS', 
    'N_val': 'N_CLS', 
    'N_test': 'N_CLS'
}

column_gaussian_label_dict = {
    'A_train': 'A_GAUSSIAN_CLS',
    'A_val': 'A_GAUSSIAN_CLS',
    'A_test': 'A_GAUSSIAN_CLS',
    'T_train': 'T_GAUSSIAN_CLS', 
    'T_val': 'T_GAUSSIAN_CLS', 
    'T_test': 'T_GAUSSIAN_CLS',
    'N_train': 'N_GAUSSIAN_CLS', 
    'N_val': 'N_GAUSSIAN_CLS', 
    'N_test': 'N_GAUSSIAN_CLS'
}

covariate_columns = [
    'Age',
    'Sex_F', 
    'Sex_M',
    'APOE A1_2', 
    'APOE A1_3', 
    'APOE A1_4', 
    'APOE A2_2',
    'APOE A2_3', 
    'APOE A2_4', 
    'LEFT_HIPPOCAMPUS_VOLUME',
    'RIGHT_HIPPOCAMPUS_VOLUME', 
    'MMSE Total Score',
    'ADAS13',
    'AD', 
    'CN', 
    'EMCI', 
    'LMCI', 
    'MCI',
    'SMC'
]

def rotate_img(volume, random_rotate_angles):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # pick angles at random
        angle = random.choice(random_rotate_angles)
        # rotate volume
        volume = ndimage.rotate(volume.astype("float32"), angle, axes=(0,2), reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    # augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    augmented_volume = scipy_rotate(volume)

    return augmented_volume

class DataGenerator_Gaussian_Labels(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self, 
        target, 
        batch_size, 
        dim, 
        shuffle,
        use_random_rotate,
        random_rotate_angles
    ):
        
        # Get data dir and subject ids
        self.data_dir = Path('../data/matched_images/{}/mri_final/{}_input_{}x{}x{}_trimmed'.format(
            target, target, dim[0], dim[1], dim[2]
        ))
        self.subject_ids = [int(x.split('.npy')[0]) for x in os.listdir(self.data_dir) if '.npy' in x]
        
        # Get labels and covariates
        label_df = pd.read_csv('../csv/generated/{}_complete_updated_gaussian.csv'.format(target))
        self.covariates = label_df[covariate_columns].T.to_dict('list')
        self.labels = label_df[column_gaussian_label_dict[target]].to_dict()
    
        self.dim = dim
        self.batch_size = batch_size
        self.use_random_rotate = use_random_rotate
        self.random_rotate_angles = random_rotate_angles
    
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.subject_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        temp_subject_ids = [self.subject_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(temp_subject_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.subject_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_subject_ids):
        # Initialization
        X_imgs = np.empty((self.batch_size, *self.dim))
        X_covariates = np.empty((self.batch_size, len(covariate_columns)), dtype=float)
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(temp_subject_ids):
            img = np.load(self.data_dir / '{:04d}.npy'.format(ID))
            if self.use_random_rotate:
                img = rotate_img(img, self.random_rotate_angles)
            
            # Store sample
            X_imgs[i,] = img
            X_covariates[i] = self.covariates[ID]

            # Store class
            y[i] = self.labels[ID]

        return (X_imgs, X_covariates), y
    
    
class DataGenerator_Gaussian_Labels_Image_Only(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self, 
        target, 
        batch_size, 
        dim, 
        shuffle,
        use_random_rotate,
        random_rotate_angles
    ):
        
        # Get data dir and subject ids
        self.data_dir = Path('../data/matched_images/{}/mri_final/{}_input_{}x{}x{}_trimmed'.format(
            target, target, dim[0], dim[1], dim[2]
        ))
        self.subject_ids = [int(x.split('.npy')[0]) for x in os.listdir(self.data_dir) if '.npy' in x]
        
        # Get labels and covariates
        label_df = pd.read_csv('../csv/generated/{}_complete_updated_gaussian.csv'.format(target))
        self.labels = label_df[column_gaussian_label_dict[target]].to_dict()
    
        self.dim = dim
        self.batch_size = batch_size
        self.use_random_rotate = use_random_rotate
        self.random_rotate_angles = random_rotate_angles
    
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.subject_ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        temp_subject_ids = [self.subject_ids[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(temp_subject_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.subject_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_subject_ids):
        # Initialization
        X_imgs = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(temp_subject_ids):
            img = np.load(self.data_dir / '{:04d}.npy'.format(ID))
            if self.use_random_rotate:
                img = rotate_img(img, self.random_rotate_angles)
            
            # Store sample
            X_imgs[i,] = img

            # Store class
            y[i] = self.labels[ID]

        return X_imgs, y