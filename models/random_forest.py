import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
from .base_model import BaseModel
import config

class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        """
        Initialise le modèle Random Forest.
        
        Args:
            **kwargs: Arguments supplémentaires pour le modèle RandomForestClassifier
        """
        self.model = RandomForestClassifier(**{**config.MODEL_CONFIG['random_forest'], **kwargs})
        self.input_shape = None
    
    def build(self, input_shape):
        """
        Configure la forme d'entrée du modèle.
        
        Args:
            input_shape: Tuple (hauteur, largeur, canaux) des images d'entrée
        """
        self.input_shape = input_shape
        # Pour le Random Forest, on a besoin d'aplatir les images
        self.n_features = input_shape[0] * input_shape[1] * input_shape[2]
    
    def _preprocess_data(self, generator):
        """
        Prétraite les données pour le Random Forest.
        
        Args:
            generator: Générateur de données Keras
            
        Returns:
            tuple: (X, y) données et étiquettes
        """
        X, y = [], []
        for i in range(len(generator)):
            batch_x, batch_y = generator[i]
            X.append(batch_x)
            y.append(batch_y)
        
        X = np.vstack(X)
        X = X.reshape(X.shape[0], -1)  # Aplatir les images
        y = np.concatenate(y)
        
        return X, y
    
    def train(self, train_generator, validation_generator, class_weights=None):
        """
        Entraîne le modèle Random Forest.
        
        Args:
            train_generator: Générateur de données d'entraînement
            validation_generator: Générateur de données de validation
            class_weights: Poids des classes pour gérer le déséquilibre
        """
        if self.input_shape is None:
            self.build(train_generator.image_shape)
        
        # Prétraiter les données d'entraînement
        X_train, y_train = self._preprocess_data(train_generator)
        
        # Entraîner le modèle avec les poids de classe si fournis
        if class_weights is not None:
            self.model.class_weight = class_weights
        
        self.model.fit(X_train, y_train)
        
        # Évaluer sur l'ensemble de validation
        if validation_generator is not None:
            X_val, y_val = self._preprocess_data(validation_generator)
            y_pred = self.model.predict(X_val)
            y_proba = self.model.predict_proba(X_val)[:, 1]
            
            print("\nÉvaluation sur l'ensemble de validation:")
            print(classification_report(y_val, y_pred, target_names=config.CLASSES))
            print(f"AUC-ROC: {roc_auc_score(y_val, y_proba):.4f}")
    
    def evaluate(self, test_generator):
        """
        Évalue le modèle sur l'ensemble de test.
        
        Args:
            test_generator: Générateur de données de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        X_test, y_test = self._preprocess_data(test_generator)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calcul des métriques
        accuracy = accuracy_score(y_test, y_pred)
        auc_roc = roc_auc_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, target_names=config.CLASSES, output_dict=True)
        
        print("\nÉvaluation sur l'ensemble de test:")
        print(f"Exactitude: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(classification_report(y_test, y_pred, target_names=config.CLASSES))
        
        # Afficher l'importance des caractéristiques
        self._display_feature_importances()
        
        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'report': report
        }
    
    def _display_feature_importances(self, top_n=20):
        """Affiche les caractéristiques les plus importantes."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            print("\nTop 20 des caractéristiques les plus importantes:")
            for i in indices:
                print(f"Caractéristique {i}: {importances[i]:.6f}")
    
    def predict(self, X):
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X: Données d'entrée (peut être une image ou un tableau d'images)
            
        Returns:
            Prédictions du modèle
        """
        if len(X.shape) > 2:  # Si c'est une image ou un batch d'images
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
            
        return self.model.predict(X_flat)
    
    def predict_proba(self, X):
        """
        Retourne les probabilités pour chaque classe.
        
        Args:
            X: Données d'entrée
            
        Returns:
            Probabilités pour chaque classe
        """
        if len(X.shape) > 2:  # Si c'est une image ou un batch d'images
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
            
        return self.model.predict_proba(X_flat)
    
    def save(self, path):
        """
        Sauvegarde le modèle.
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'input_shape': self.input_shape
        }, path)
    
    @classmethod
    def load(cls, path):
        """
        Charge un modèle sauvegardé.
        
        Args:
            path: Chemin vers le modèle sauvegardé
            
        Returns:
            Modèle chargé
        """
        data = joblib.load(path)
        model = cls()
        model.model = data['model']
        model.input_shape = data['input_shape']
        return model
