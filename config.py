from pathlib import Path

# Chemins des données
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "chest_Xray"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Paramètres des images
IMG_HEIGHT = 150
IMG_WIDTH = 150
CHANNELS = 3  # RGB
BATCH_SIZE = 32

# Paramètres d'entraînement
EPOCHS = 20
LEARNING_RATE = 0.001

# Classes
CLASSES = ['NORMAL', 'PNEUMONIA']

# Répartition des données
TRAIN_SIZE = 0.75  # 75% pour l'entraînement
VAL_SIZE = 0.25     # 25% pour la validation

# Paramètres des modèles
MODEL_CONFIG = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs'
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'cnn': {
        'filters': [32, 64],
        'dense_units': 128,
        'dropout_rate': 0.5
    }
}
