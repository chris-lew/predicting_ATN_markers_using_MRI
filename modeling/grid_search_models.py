import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np 
import argparse

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.losses
import tensorflow.keras.metrics
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
import gc

from covariate_models import *
from build_datasets_covariate import *
from experiments_covariate import experiment_dict

# Reduce all the messages that get displayed
tf.get_logger().setLevel('INFO')


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('lr', type=float, help='learning rate')
parser.add_argument('dropout', type=float, help='dropout rate')
parser.add_argument('imgfeatures', type=int, help='img features')
parser.add_argument('gpu', help='which gpu to use')

args = parser.parse_args()

output_dir = '../data/saved_models_v2/HPO_052722/'
target = 'N'
data_dim = [182, 182, 218]

# Get experiment hyperparameters
random_rotate_angles = [-20, -10, -5, 5, 10, 20]

# We can keep this consistent for things for that aren't going to change
batch_size = 16
epoch_count = 20
decay = 100000

learning_rate = args.lr
dropout = args.dropout
image_features = args.imgfeatures

learning_rate_list = [3e-2, 3e-3,3e-4, 3e-5]
dropout_list = [0, 0.3, 0.4, 0.5, 0.6]
image_features_list = [50, 25, 100]

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

physical_devices_gpu = tf.config.list_physical_devices('GPU')
print("Number of GPUs:", len(physical_devices_gpu))

physical_devices_cpu = tf.config.list_physical_devices('CPU')
print("Number of CPUs:", len(physical_devices_cpu))

for gpu in physical_devices_gpu:
    tf.config.experimental.set_memory_growth(gpu, True)
 
output_str = '_gaussian_labels_lr{}_dropout{}_imagefeatures{}'.format(
    learning_rate, dropout, image_features
)

print('#'*40)
print(output_str)
print('#'*40)

# We'll reinitialize things each time just in case

strategy = tf.distribute.MirroredStrategy(
    ['GPU:{}'.format(x) for x in range(len(physical_devices_gpu))]
)

# Set up data generators
train_gen = DataGenerator_Gaussian_Labels(
    target = target + '_train', 
    batch_size = batch_size, 
    dim = data_dim, 
    shuffle = True,
    use_random_rotate = True,
    random_rotate_angles = random_rotate_angles
)
val_gen = DataGenerator_Gaussian_Labels(
    target = target + '_val', 
    batch_size = batch_size, 
    dim = data_dim, 
    shuffle = True,
    use_random_rotate = False,
    random_rotate_angles = random_rotate_angles
)

# Learning rate schedule
lr_schedule = schedules.ExponentialDecay(
    learning_rate, decay_steps=decay, decay_rate=0.96, staircase=True
)

# Build and compile
with strategy.scope():
    network = AD_classifier_plus_tuning(
        width=data_dim[0], height=data_dim[1], depth=data_dim[2],
        dropout=dropout,
        image_features=image_features) 
    network.compile(
        optimizer = Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.binary_crossentropy, 
        metrics=['binary_accuracy']
    )

    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        output_dir+'{}.h5'.format(output_str), save_best_only=True
    )

    training_history = network.fit(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch = len(train_gen),
        validation_steps = len(val_gen),
        epochs=epoch_count,
        callbacks=[checkpoint_cb],
        verbose = 2,
        use_multiprocessing=True,
        workers=6
    )

    np.save(output_dir+'{}_history.npy'.format(output_str), training_history.history)


