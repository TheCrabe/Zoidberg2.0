import argparse
import os
import json
from datetime import datetime
from data_loader import load_data, get_class_weights
from models import LogisticRegressionModel, RandomForestModel, CNNModel
import config

def get_model(model_name):
    """
    Renvoie une instance du modèle demandé.
    
    Args:
        model_name: Nom du modèle ('logistic_regression', 'random_forest' ou 'cnn')
        
    Returns:
        Instance du modèle
    """
    if model_name == 'logistic_regression':
        return LogisticRegressionModel()
    elif model_name == 'random_forest':
        return RandomForestModel()
    elif model_name == 'cnn':
        return CNNModel()
    else:
        raise ValueError(f"Modèle non pris en charge: {model_name}. Choisissez parmi: logistic_regression, random_forest, cnn")

def main():
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Entraînement et évaluation de modèles de classification d\'images médicales')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['logistic_regression', 'random_forest', 'cnn'],
                        help='Modèle à entraîner')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Nombre d\'époques pour l\'entraînement')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Taille des lots pour l\'entraînement')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Répertoire pour sauvegarder les résultats')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = os.path.join(args.output_dir, f"{args.model}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Chargement des données...")
    train_generator, val_generator, test_generator = load_data()
    
    # Afficher des informations sur les données
    print("\nInformations sur les données:")
    print(f"- Nombre d'images d'entraînement: {train_generator.samples}")
    print(f"- Nombre d'images de validation: {val_generator.samples}")
    print(f"- Nombre d'images de test: {test_generator.samples}")
    print(f"- Taille des images: {train_generator.image_shape}")
    
    # Calculer les poids des classes pour gérer le déséquilibre
    class_weights = get_class_weights(train_generator)
    print(f"\nPoids des classes: {class_weights}")
    
    # Initialiser et construire le modèle
    print(f"\nInitialisation du modèle {args.model}...")
    model = get_model(args.model)
    model.build(train_generator.image_shape)
    
    # Entraîner le modèle
    print(f"\nDébut de l'entraînement du modèle {args.model}...")
    history = model.train(train_generator, val_generator, class_weights)
    
    # Évaluer le modèle sur l'ensemble de test
    print("\nÉvaluation sur l'ensemble de test...")
    metrics = model.evaluate(test_generator)
    
    # Sauvegarder le modèle et les métriques
    print(f"\nSauvegarde du modèle et des résultats dans {model_dir}")
    model_path = os.path.join(model_dir, f"{args.model}_model")
    model.save(model_path)
    
    # Sauvegarder les métriques
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Sauvegarder la configuration
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        config_dict = {
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'image_height': config.IMG_HEIGHT,
            'image_width': config.IMG_WIDTH,
            'channels': config.CHANNELS,
            'learning_rate': config.LEARNING_RATE,
            'class_weights': class_weights,
            'timestamp': timestamp
        }
        json.dump(config_dict, f, indent=4)
    
    print("\nEntraînement et évaluation terminés avec succès!")

if __name__ == "__main__":
    main()
