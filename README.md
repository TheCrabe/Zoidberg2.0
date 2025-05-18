# Classification d'images médicales : Pneumonie

Ce projet implémente différents modèles d'apprentissage automatique pour la classification d'images de radiographies pulmonaires afin de détecter les cas de pneumonie.

## Structure du projet

```
.
├── chest_Xray/                  # Dossier contenant les données (non inclus dans le dépôt)
│   ├── train/                   # Données d'entraînement
│   │   ├── NORMAL/              # Images normales
│   │   └── PNEUMONIA/           # Images avec pneumonie
│   ├── val/                     # Données de validation
│   └── test/                    # Données de test
├── models/                      # Implémentations des modèles
│   ├── __init__.py
│   ├── base_model.py            # Classe de base abstraite
│   ├── logistic_regression.py   # Régression logistique
│   ├── random_forest.py         # Forêt aléatoire
│   └── cnn.py                   # Réseau de neurones convolutif
├── config.py                    # Configuration du projet
├── data_loader.py               # Chargement et prétraitement des données
├── train.py                     # Script d'entraînement
├── predict.py                   # Script de prédiction
└── requirements.txt             # Dépendances Python
```

## Installation

1. Clonez le dépôt :
   ```bash
   git clone [URL_DU_DEPOT]
   cd Zoidberg2.0
   ```

2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

3. Téléchargez le jeu de données et placez-le dans le dossier `chest_Xray/` avec la structure indiquée ci-dessus.

## Utilisation

### Entraînement d'un modèle

Pour entraîner un modèle, utilisez le script `train.py` :

```bash
python train.py --model [MODEL_TYPE] --epochs [N_EPOCHS] --batch_size [BATCH_SIZE] --output_dir [OUTPUT_DIR]
```

**Options :**
- `--model` : Type de modèle à entraîner (`logistic_regression`, `random_forest` ou `cnn`)
- `--epochs` : Nombre d'époques d'entraînement (par défaut : 20)
- `--batch_size` : Taille des lots d'entraînement (par défaut : 32)
- `--output_dir` : Répertoire de sortie pour sauvegarder les résultats (par défaut : 'results')

**Exemple :**
```bash
python train.py --model cnn --epochs 30 --batch_size 16 --output_dir results/cnn_model
```

### Faire des prédictions

Pour faire des prédictions avec un modèle entraîné, utilisez le script `predict.py` :

```bash
python predict.py --model_path [CHEMIN_VERS_MODELE] --model_type [TYPE_MODELE] --image_path [CHEMIN_IMAGE]
```

**Options :**
- `--model_path` : Chemin vers le modèle sauvegardé
- `--model_type` : Type de modèle (`logistic_regression`, `random_forest` ou `cnn`)
- `--image_path` : Chemin vers l'image à prédire

**Exemple :**
```bash
python predict.py --model_path results/cnn_model/cnn_model --model_type cnn --image_path chest_Xray/test/NORMAL/IM-0001-0001.jpeg
```

## Modèles implémentés

1. **Régression logistique** : Un modèle linéaire simple pour la classification binaire.
2. **Forêt aléatoire** : Un ensemble d'arbres de décision qui améliore la précision et contrôle le surajustement.
3. **Réseau de neurones convolutif (CNN)** : Un modèle profond spécialement conçu pour le traitement d'images.

## Structure du code

Le code est organisé de manière modulaire pour faciliter l'ajout de nouveaux modèles :

- `base_model.py` : Définit une interface commune pour tous les modèles.
- Chaque modèle implémente les méthodes définies dans `BaseModel`.
- `data_loader.py` : Gère le chargement et le prétraitement des données.
- `config.py` : Contient les paramètres globaux du projet.

## Améliorations possibles

1. **Augmentation des données** : Ajouter plus de techniques d'augmentation pour améliorer la généralisation.
2. **Modèles plus avancés** : Implémenter des architectures CNN plus complexes (ResNet, DenseNet, etc.).
3. **Explicabilité** : Ajouter des visualisations pour comprendre les décisions du modèle (Grad-CAM, SHAP, etc.).
4. **Déploiement** : Créer une API REST pour servir le modèle.

## Auteur

[Votre nom]

## Licence

Ce projet est sous licence [MIT](LICENSE).
