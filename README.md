# Système de Prédiction du Risque Cardiaque

Application Streamlit pour la démonstration des modèles de prédiction de risque cardiaque.

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Étape 1 : Entraîner les modèles

D'abord, entraînez et sauvegardez les modèles :
```bash
python train_models.py
```

Cette étape va créer les fichiers suivants :
- `model_BO1.pkl` - Modèle de classification binaire
- `model_BO2.pkl` - Modèle de régression
- `model_BO3.pkl` - Modèle de classification multi-classe
- `model_BO4.pkl` - Modèle de clustering
- `scaler_BO1.pkl`, `scaler_BO2.pkl`, `scaler_BO3.pkl`, `scaler_BO4.pkl` - Scalers
- `features_BO1.pkl`, `features_BO2.pkl`, `features_BO3.pkl`, `features_BO4.pkl` - Features
- `top_5_features_BO3.pkl` - Top 5 features pour BO3
- `cluster_info_BO4.csv` - Informations sur les clusters

### Étape 2 : Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : `http://localhost:8501`

## Structure de l'Application

- **Page d'Accueil** : Présentation du projet, objectifs métier et pipeline
- **BO1** : Classification binaire - Identification des patients à haut risque
- **BO2** : Score continu - Prédiction d'un score de risque (0-100)
- **BO3** : Classification multi-classe - Classement en 3 niveaux (Faible/Moyen/Élevé)
- **BO4** : Clustering - Attribution à un groupe de patients similaires

## Objectifs Métier

- **BO1** : Identifier précocement les patients asymptomatiques à haut risque
- **BO2** : Évaluer le risque comportemental par un score continu
- **BO3** : Identifier un score de risque basé sur les facteurs les plus prédictifs
- **BO4** : Identifier des groupes de patients similaires

