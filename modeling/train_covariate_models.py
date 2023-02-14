import os
import numpy as np 
import argparse

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.losses
import tensorflow.keras.metrics
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as k

from covariate_models import *
from build_datasets_covariate import *
from experiments_covariate import experiment_dict

# Reduce all the messages that get displayed
tf.get_logger().setLevel('INFO')

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('experiment', help='name of the experiment in expierments.py to run')
parser.add_argument('gpu', help='which gpu to use')
parser.add_argument('--load', type=bool, help='if you want to load the model instead of training a new one')
args = parser.parse_args()
experiment_name = args.experiment

# Get experiment hyperparameters
target = experiment_dict[experiment_name]['target']
data_dim = experiment_dict[experiment_name]['data_dim']
random_rotate_angles = [-20, -10, -5, 5, 10, 20]

# Grab missing hyperparameters if missing
if 'batch_size' in experiment_dict[experiment_name]:
    batch_size = experiment_dict[experiment_name]['batch_size']
else:
    batch_size = 16

if 'epoch_count' in experiment_dict[experiment_name]:
    epoch_count = experiment_dict[experiment_name]['epoch_count']
else:
    epoch_count = 50

if 'patience' in experiment_dict[experiment_name]:
    patience = experiment_dict[experiment_name]['patience']
else:
    patience = 10

if 'learning_rate' in experiment_dict[experiment_name]:
    learning_rate = experiment_dict[experiment_name]['learning_rate']
else:
    learning_rate = 1e-4

if 'decay' in experiment_dict[experiment_name]:
    decay = experiment_dict[experiment_name]['decay']
else:
    decay = 100000

# Set up GPUs in mirrored strategy 
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

physical_devices_gpu = tf.config.list_physical_devices('GPU')
print("Number of GPUs:", len(physical_devices_gpu))

physical_devices_cpu = tf.config.list_physical_devices('CPU')
print("Number of CPUs:", len(physical_devices_cpu))

for gpu in physical_devices_gpu:
    tf.config.experimental.set_memory_growth(gpu, True)

strategy = tf.distribute.MirroredStrategy(
    ['GPU:{}'.format(x) for x in range(len(physical_devices_gpu))]
)

# Set up data generators
train_gen = DataGenerator(
    target = target + '_train', 
    batch_size = batch_size, 
    dim = data_dim, 
    shuffle = True,
    use_random_rotate = True,
    random_rotate_angles = random_rotate_angles
)
val_gen = DataGenerator(
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

# Build and compile website
with strategy.scope():
    network = AD_classifier_plus(width=data_dim[0], height=data_dim[1], depth=data_dim[2]) 
    network.compile(
        optimizer = Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.binary_crossentropy, 
        metrics=['binary_accuracy']
    )

network.summary()

# This callback will stop the training when there is no improvement in the validation loss 
# for patience # of consecutive epochs.  
earlystop = EarlyStopping(monitor='loss', 
                          patience=patience, 
                          verbose=1, 
                          mode='min')

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    '../data/saved_models_v2/{}.h5'.format(experiment_name), save_best_only=True
)
        
if args.load:
    saved_model = '../data/saved_models_v2/{}.h5'.format(experiment_name)
    network.load_weights(saved_model)

training_history = network.fit(
    train_gen,
    validation_data=val_gen,
    steps_per_epoch = len(train_gen),
    validation_steps = len(val_gen),
    epochs=epoch_count,
    callbacks=[earlystop, checkpoint_cb],
    verbose = 1,
    use_multiprocessing=True,
    workers=6
)

np.save('../data/saved_models_v2/{}_history.npy'.format(experiment_name), training_history.history)