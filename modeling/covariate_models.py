import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


def AD_classifier_plus_tuning(width=182, height=182, depth=218, dropout=0.5, image_features=50):
    image_input = layers.Input((width, height, depth, 1))
    feature_input = layers.Input((19,))

    x = layers.Conv3D(filters=8, kernel_size=3, padding="same")(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    
    x = layers.Conv3D(filters=16, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(1300, activation='relu')(x)
    x = layers.Dense(image_features, activation='relu')(x)
    
    x = layers.Concatenate()([x, feature_input])
    x = layers.Dense(25, activation='relu')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Define the model.
    model = keras.Model([image_input, feature_input], outputs)
    return model

def AD_classifier_plus_tuning_img_only(width=182, height=182, depth=218, dropout=0.5, image_features=50):
    image_input = layers.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=8, kernel_size=3, padding="same")(image_input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    
    x = layers.Conv3D(filters=16, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=32, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=64, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(1300, activation='relu')(x)
    x = layers.Dense(image_features, activation='relu')(x)
    
    x = layers.Dense(25, activation='relu')(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    # Define the model.
    model = keras.Model(image_input, outputs)
    return model