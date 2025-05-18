import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config

def load_data():
    """
    Charge les données d'images et divise en ensembles d'entraînement et de validation.
    """
    # Créer un générateur d'images avec augmentation de données pour l'entraînement
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=config.VAL_SIZE
    )

    # Chargement des données d'entraînement et de validation
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,  # On utilise le même répertoire mais avec subset='validation'
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    # Chargement des données de test
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator

def get_class_weights(generator):
    """
    Calcule les poids des classes pour gérer le déséquilibre des classes.
    """
    class_counts = np.bincount(generator.classes)
    total = len(generator.classes)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    return class_weights
