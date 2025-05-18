import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from .base_model import BaseModel
import config

class CNNModel(BaseModel):
    def __init__(self, **kwargs):
        """
        Initialise le modèle CNN.
        
        Args:
            **kwargs: Arguments supplémentaires pour la configuration du modèle
        """
        self.model = None
        self.input_shape = None
        self.config = {**config.MODEL_CONFIG['cnn'], **kwargs}
    
    def build(self, input_shape):
        """
        Construit l'architecture du modèle CNN.
        
        Args:
            input_shape: Tuple (hauteur, largeur, canaux) des images d'entrée
        """
        self.input_shape = input_shape
        
        model = Sequential()
        
        # Couche de convolution 1
        model.add(Conv2D(
            filters=self.config['filters'][0],
            kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape,
            padding='same'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Couche de convolution 2
        model.add(Conv2D(
            filters=self.config['filters'][1],
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Aplatir les caractéristiques pour les couches denses
        model.add(Flatten())
        
        # Couche dense avec dropout pour la régularisation
        model.add(Dense(self.config['dense_units'], activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Couche de sortie (classification binaire)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compiler le modèle
        model.compile(
            optimizer=Adam(learning_rate=config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("Résumé du modèle CNN:")
        model.summary()
    
    def train(self, train_generator, validation_generator, class_weights=None):
        """
        Entraîne le modèle CNN.
        
        Args:
            train_generator: Générateur de données d'entraînement
            validation_generator: Générateur de données de validation
            class_weights: Poids des classes pour gérer le déséquilibre
        """
        if self.model is None:
            self.build(train_generator.image_shape)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                filepath='best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Entraînement du modèle
        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=config.EPOCHS,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        return history
    
    def evaluate(self, test_generator):
        """
        Évalue le modèle sur l'ensemble de test.
        
        Args:
            test_generator: Générateur de données de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        # Évaluation sur l'ensemble de test
        test_loss, test_accuracy = self.model.evaluate(test_generator)
        
        # Prédictions pour le rapport de classification
        y_pred = (self.model.predict(test_generator) > 0.5).astype(int)
        y_true = test_generator.classes
        
        # Calcul du score AUC-ROC
        y_proba = self.model.predict(test_generator)
        auc_roc = roc_auc_score(y_true, y_proba)
        
        # Rapport de classification
        report = classification_report(y_true, y_pred, target_names=config.CLASSES, output_dict=True)
        
        print("\nÉvaluation sur l'ensemble de test:")
        print(f"Perte: {test_loss:.4f}")
        print(f"Exactitude: {test_accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(classification_report(y_true, y_pred, target_names=config.CLASSES))
        
        return {
            'loss': test_loss,
            'accuracy': test_accuracy,
            'auc_roc': auc_roc,
            'report': report
        }
    
    def predict(self, X):
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X: Données d'entrée (peut être une image ou un tableau d'images)
            
        Returns:
            Prédictions du modèle (0 ou 1)
        """
        # S'assurer que les données sont au bon format
        if len(X.shape) == 3:  # Une seule image
            X = np.expand_dims(X, axis=0)
        
        # Faire la prédiction
        predictions = self.model.predict(X)
        
        # Convertir les probabilités en classes (0 ou 1)
        return (predictions > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque classe.
        
        Args:
            X: Données d'entrée
            
        Returns:
            Probabilités pour chaque classe
        """
        # S'assurer que les données sont au bon format
        if len(X.shape) == 3:  # Une seule image
            X = np.expand_dims(X, axis=0)
            
        # Obtenir les probabilités
        probas = self.model.predict(X)
        
        # Retourner les probabilités pour les deux classes
        return np.column_stack((1 - probas, probas))
    
    def save(self, path):
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_model(self.model, path)
    
    @classmethod
    def load(cls, path):
        """
        Charge un modèle sauvegardé.
        
        Args:
            path: Chemin vers le modèle sauvegardé
            
        Returns:
            Modèle chargé
        """
        model = cls()
        model.model = load_model(path)
        model.input_shape = model.model.input_shape[1:]
        return model
