import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from models import LogisticRegressionModel, RandomForestModel, CNNModel
import config

def load_image(image_path, target_size):
    """
    Charge et prétraite une image pour la prédiction.
    
    Args:
        image_path: Chemin vers l'image
        target_size: Tuple (hauteur, largeur) pour le redimensionnement
        
    Returns:
        Image prétraitée
    """
    # Charger l'image
    img = Image.open(image_path).convert('RGB')
    
    # Redimensionner l'image
    img = img.resize((target_size[1], target_size[0]))
    
    # Convertir en tableau numpy et normaliser
    img_array = np.array(img) / 255.0
    
    return img_array

def load_model(model_path, model_type):
    """
    Charge un modèle sauvegardé.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        model_type: Type de modèle ('logistic_regression', 'random_forest' ou 'cnn')
        
    Returns:
        Modèle chargé
    """
    if model_type == 'logistic_regression':
        model = LogisticRegressionModel()
    elif model_type == 'random_forest':
        model = RandomForestModel()
    elif model_type == 'cnn':
        model = CNNModel()
    else:
        raise ValueError(f"Type de modèle non pris en charge: {model_type}")
    
    # Charger les poids du modèle
    return model.load(model_path)

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Faire des prédictions avec un modèle entraîné')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Chemin vers le modèle sauvegardé')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['logistic_regression', 'random_forest', 'cnn'],
                        help='Type de modèle')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Chemin vers l\'image à prédire')
    
    args = parser.parse_args()
    
    # Vérifier que le fichier image existe
    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Le fichier {args.image_path} n'existe pas.")
    
    # Charger le modèle
    print(f"Chargement du modèle {args.model_type} depuis {args.model_path}...")
    model = load_model(args.model_path, args.model_type)
    
    # Charger et prétraiter l'image
    print(f"Chargement de l'image {args.image_path}...")
    img_array = load_image(args.image_path, (config.IMG_HEIGHT, config.IMG_WIDTH))
    
    # Faire la prédiction
    print("\nPrédiction en cours...")
    
    # Pour les modèles qui supportent predict_proba
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(np.expand_dims(img_array, axis=0))[0]
        predicted_class = probas.argmax()
        confidence = probas[predicted_class]
    else:
        # Pour les modèles qui n'ont que predict
        prediction = model.predict(np.expand_dims(img_array, axis=0))[0]
        predicted_class = prediction
        confidence = 1.0  # Confiance à 100% si pas de probabilités
    
    # Afficher les résultats
    print("\nRésultats de la prédiction:")
    print(f"- Classe prédite: {config.CLASSES[predicted_class]} (Classe {predicted_class})")
    
    if hasattr(model, 'predict_proba'):
        print("\nProbabilités par classe:")
        for i, class_name in enumerate(config.CLASSES):
            print(f"  {class_name}: {probas[i]:.4f}")
    
    print(f"\nConfiance: {confidence*100:.2f}%")

if __name__ == "__main__":
    main()
