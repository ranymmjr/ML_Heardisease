"""
Script pour entraîner et sauvegarder les modèles pour chaque objectif métier
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
import joblib
import os

# Chargement des données
print("Chargement des données...")
data = pd.read_csv('heart_disease.csv', na_values=['NA', 'nan', 'NaN', ''])

# Nettoyage des données
print("Nettoyage des données...")

# Imputation des valeurs manquantes
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
numeric_cols = ['cigsPerDay', 'BPMeds', 'totChol', 'BMI', 'heartRate', 'glucose', 'age', 'sysBP', 'diaBP']
for col in numeric_cols:
    if col in data.columns:
        data[col] = imputer.fit_transform(data[[col]]).ravel()

# Remplacer les valeurs manquantes pour les colonnes numériques
numeric_cols_all = data.select_dtypes(include=[np.number]).columns
for col in numeric_cols_all:
    if data[col].isna().any():
        data[col].fillna(data[col].median(), inplace=True)

# ========== BO1 : Classification binaire (Haut risque) ==========
print("\n=== BO1 : Classification binaire ===")
features_BO1 = ['age', 'sysBP', 'diaBP', 'totChol', 'BMI', 'heartRate', 'glucose', 
                'currentSmoker', 'prevalentHyp', 'diabetes']
X_BO1 = data[features_BO1].fillna(data[features_BO1].median())

# Création de la variable cible : haut risque basé sur Heart_ stroke
data['high_risk'] = (data['Heart_ stroke'].str.lower().str.strip() == 'yes').astype(int)
y_BO1 = data['high_risk']

X_train_BO1, X_test_BO1, y_train_BO1, y_test_BO1 = train_test_split(
    X_BO1, y_BO1, test_size=0.3, random_state=42, stratify=y_BO1
)

scaler_BO1 = StandardScaler()
X_train_BO1_scaled = scaler_BO1.fit_transform(X_train_BO1)
X_test_BO1_scaled = scaler_BO1.transform(X_test_BO1)

# Entraînement XGBoost pour BO1
print("Entraînement XGBoost pour BO1...")
xgb_BO1 = XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
param_grid_BO1 = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
grid_BO1 = GridSearchCV(xgb_BO1, param_grid_BO1, cv=3, scoring='f1', n_jobs=-1, verbose=0)
grid_BO1.fit(X_train_BO1_scaled, y_train_BO1)
model_BO1 = grid_BO1.best_estimator_

# Sauvegarde
joblib.dump(model_BO1, 'model_BO1.pkl')
joblib.dump(scaler_BO1, 'scaler_BO1.pkl')
print(f"BO1 - F1 Score: {f1_score(y_test_BO1, model_BO1.predict(X_test_BO1_scaled)):.4f}")

# ========== BO2 : Régression (Score de risque continu) ==========
print("\n=== BO2 : Régression ===")
features_BO2 = ['age', 'sysBP', 'totChol', 'BMI', 'glucose']
X_BO2 = data[features_BO2].fillna(data[features_BO2].median())

# Création du score de risque continu
# S'assurer que toutes les colonnes nécessaires sont remplies
for col in ['age', 'sysBP', 'totChol', 'BMI', 'glucose', 'currentSmoker', 'prevalentHyp', 'diabetes']:
    if col in data.columns and data[col].isna().any():
        if data[col].dtype in ['int64', 'float64']:
            data[col].fillna(data[col].median(), inplace=True)
        else:
            data[col].fillna(0, inplace=True)

data['cardiac_risk_score'] = (
    data['age'] * 0.1 +
    data['sysBP'] * 0.3 +
    data['totChol'] * 0.2 +
    data['BMI'] * 0.2 +
    data['glucose'] * 0.2 +
    data['currentSmoker'] * 10 +
    data['prevalentHyp'] * 15 +
    data['diabetes'] * 20
)
# Normalisation entre 0 et 100
score_min = data['cardiac_risk_score'].min()
score_max = data['cardiac_risk_score'].max()
if score_max > score_min:
    data['cardiac_risk_score'] = ((data['cardiac_risk_score'] - score_min) / 
                                  (score_max - score_min) * 100)
else:
    data['cardiac_risk_score'] = 50  # Valeur par défaut si toutes les valeurs sont identiques

# S'assurer qu'il n'y a pas de NaN
data['cardiac_risk_score'] = data['cardiac_risk_score'].fillna(50)

y_BO2 = data['cardiac_risk_score']

X_train_BO2, X_test_BO2, y_train_BO2, y_test_BO2 = train_test_split(
    X_BO2, y_BO2, test_size=0.3, random_state=42
)

scaler_BO2 = StandardScaler()
X_train_BO2_scaled = scaler_BO2.fit_transform(X_train_BO2)
X_test_BO2_scaled = scaler_BO2.transform(X_test_BO2)

# Entraînement XGBoost Regressor pour BO2
print("Entraînement XGBoost Regressor pour BO2...")
xgb_BO2 = XGBRegressor(random_state=42, n_jobs=-1)
param_grid_BO2 = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}
grid_BO2 = GridSearchCV(xgb_BO2, param_grid_BO2, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
grid_BO2.fit(X_train_BO2_scaled, y_train_BO2)
model_BO2 = grid_BO2.best_estimator_

# Sauvegarde
joblib.dump(model_BO2, 'model_BO2.pkl')
joblib.dump(scaler_BO2, 'scaler_BO2.pkl')
print(f"BO2 - R² Score: {r2_score(y_test_BO2, model_BO2.predict(X_test_BO2_scaled)):.4f}")

# ========== BO3 : Classification multi-classe (Faible/Moyen/Élevé) ==========
print("\n=== BO3 : Classification multi-classe ===")
features_BO3 = ['age', 'sysBP', 'diaBP', 'totChol', 'BMI', 'heartRate', 'glucose', 
                'currentSmoker', 'prevalentHyp', 'diabetes']
X_BO3 = data[features_BO3].fillna(data[features_BO3].median())

# S'assurer que cardiac_risk_score n'a pas de NaN
data['cardiac_risk_score'] = data['cardiac_risk_score'].fillna(data['cardiac_risk_score'].median())

# Création des catégories de risque basées sur le score
y_BO3 = pd.cut(data['cardiac_risk_score'], bins=[0, 33, 66, 100], labels=[0, 1, 2], include_lowest=True)
# Supprimer les NaN créés par pd.cut (si le score est en dehors des bins)
y_BO3 = y_BO3.fillna(1)  # Remplacer les NaN par la catégorie moyenne
y_BO3 = y_BO3.astype(int)

X_train_BO3, X_test_BO3, y_train_BO3, y_test_BO3 = train_test_split(
    X_BO3, y_BO3, test_size=0.3, random_state=42, stratify=y_BO3
)

scaler_BO3 = StandardScaler()
X_train_BO3_scaled = scaler_BO3.fit_transform(X_train_BO3)
X_test_BO3_scaled = scaler_BO3.transform(X_test_BO3)

# Entraînement Random Forest pour BO3
print("Entraînement Random Forest pour BO3...")
rf_BO3 = RandomForestClassifier(random_state=42, n_jobs=-1)
param_grid_BO3 = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5]
}
grid_BO3 = GridSearchCV(rf_BO3, param_grid_BO3, cv=3, scoring='f1_macro', n_jobs=-1, verbose=0)
grid_BO3.fit(X_train_BO3_scaled, y_train_BO3)
model_BO3 = grid_BO3.best_estimator_

# Sauvegarde
joblib.dump(model_BO3, 'model_BO3.pkl')
joblib.dump(scaler_BO3, 'scaler_BO3.pkl')
print(f"BO3 - Accuracy: {accuracy_score(y_test_BO3, model_BO3.predict(X_test_BO3_scaled)):.4f}")

# Extraction des 5 features les plus importantes pour BO3
feature_importance = pd.DataFrame({
    'Feature': features_BO3,
    'Importance': model_BO3.feature_importances_
}).sort_values('Importance', ascending=False)
top_5_features_BO3 = feature_importance.head(5)['Feature'].tolist()
joblib.dump(top_5_features_BO3, 'top_5_features_BO3.pkl')
print(f"Top 5 features BO3: {top_5_features_BO3}")

# ========== BO4 : Clustering ==========
print("\n=== BO4 : Clustering ===")
features_BO4 = ['age', 'sysBP', 'totChol', 'BMI', 'currentSmoker', 'glucose']
X_BO4 = data[features_BO4].fillna(data[features_BO4].median())

scaler_BO4 = StandardScaler()
X_BO4_scaled = scaler_BO4.fit_transform(X_BO4)

# Entraînement KMeans pour BO4
print("Entraînement KMeans pour BO4...")
kmeans_BO4 = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_BO4.fit(X_BO4_scaled)

# Sauvegarde
joblib.dump(kmeans_BO4, 'model_BO4.pkl')
joblib.dump(scaler_BO4, 'scaler_BO4.pkl')

# Sauvegarde des informations sur les clusters
cluster_centers = scaler_BO4.inverse_transform(kmeans_BO4.cluster_centers_)
cluster_info = pd.DataFrame(cluster_centers, columns=features_BO4)
cluster_info.to_csv('cluster_info_BO4.csv', index=False)
print("BO4 - Clustering terminé avec 3 clusters")

# Sauvegarde des features pour chaque objectif
joblib.dump(features_BO1, 'features_BO1.pkl')
joblib.dump(features_BO2, 'features_BO2.pkl')
joblib.dump(features_BO3, 'features_BO3.pkl')
joblib.dump(features_BO4, 'features_BO4.pkl')

print("\n✅ Tous les modèles ont été entraînés et sauvegardés avec succès!")

