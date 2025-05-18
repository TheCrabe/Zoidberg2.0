from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Classe de base abstraite pour tous les modèles.
    Tous les modèles doivent implémenter ces méthodes.
    """
    
    @abstractmethod
    def build(self, input_shape):
        """Construit l'architecture du modèle."""
        pass
    
    @abstractmethod
    def train(self, train_generator, validation_generator, class_weights=None):
        """
        Entraîne le modèle.
        
        Args:
            train_generator: Générateur de données d'entraînement
            validation_generator: Générateur de données de validation
            class_weights: Poids des classes pour gérer le déséquilibre
        """
        pass
    
    @abstractmethod
    def evaluate(self, test_generator):
        """
        Évalue le modèle sur l'ensemble de test.
        
        Args:
            test_generator: Générateur de données de test
            
        Returns:
            dict: Métriques d'évaluation
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Effectue des prédictions sur de nouvelles données.
        
        Args:
            X: Données d'entrée
            
        Returns:
            Prédictions du modèle
        """
        pass
    
    @abstractmethod
    def save(self, path):
        """Sauvegarde le modèle au chemin spécifié."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path):
        """Charge un modèle depuis le chemin spécifié."""
        pass
