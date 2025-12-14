# Guide de Déploiement sur Streamlit Cloud

Ce guide vous explique comment déployer votre application Streamlit sur Streamlit Cloud (gratuit et hébergé).

## 📋 Prérequis

1. **Compte GitHub** : Créez un compte sur [GitHub](https://github.com) si vous n'en avez pas
2. **Compte Streamlit Cloud** : Créez un compte sur [share.streamlit.io](https://share.streamlit.io) (connexion via GitHub)
3. **Git installé** : Téléchargez [Git](https://git-scm.com/downloads) si nécessaire

## 🚀 Étapes de Déploiement

### Étape 1 : Préparer votre projet local

Assurez-vous que votre projet contient :
- ✅ `app.py` (fichier principal de l'application)
- ✅ `requirements.txt` (dépendances Python)
- ✅ Tous les fichiers `.pkl` (modèles et scalers)
- ✅ Tous les fichiers `.csv` nécessaires (comme `cluster_info_BO4.csv`)

### Étape 2 : Créer un repository GitHub

1. **Créer un nouveau repository sur GitHub** :
   - Allez sur [github.com](https://github.com)
   - Cliquez sur le bouton "+" en haut à droite → "New repository"
   - Nommez votre repository (ex: `heart-disease-prediction`)
   - Choisissez "Public" (nécessaire pour la version gratuite de Streamlit Cloud)
   - **Ne cochez PAS** "Initialize with README"
   - Cliquez sur "Create repository"

2. **Initialiser Git dans votre projet local** :
   ```bash
   cd "D:\Desktop\PROJET ML"
   git init
   git add .
   git commit -m "Initial commit: Application Streamlit de prédiction du risque cardiaque"
   ```

3. **Connecter votre projet local à GitHub** :
   ```bash
   git remote add origin https://github.com/VOTRE_USERNAME/VOTRE_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```
   
   Remplacez `VOTRE_USERNAME` et `VOTRE_REPO_NAME` par vos informations.

### Étape 3 : Déployer sur Streamlit Cloud

1. **Se connecter à Streamlit Cloud** :
   - Allez sur [share.streamlit.io](https://share.streamlit.io)
   - Cliquez sur "Sign in" et connectez-vous avec votre compte GitHub

2. **Créer une nouvelle application** :
   - Cliquez sur "New app"
   - Sélectionnez votre repository GitHub
   - Sélectionnez la branche `main`
   - Dans "Main file path", entrez : `app.py`
   - Cliquez sur "Deploy"

3. **Attendre le déploiement** :
   - Streamlit Cloud va automatiquement :
     - Installer les dépendances depuis `requirements.txt`
     - Lancer votre application
   - Le processus prend généralement 2-5 minutes

4. **Votre application est en ligne !** :
   - Une fois le déploiement terminé, vous recevrez une URL publique
   - Exemple : `https://votre-app.streamlit.app`

## 📝 Fichiers Requis

Votre repository doit contenir :

```
PROJET ML/
├── app.py                    # Application principale
├── requirements.txt          # Dépendances Python
├── train_models.py          # Script d'entraînement (optionnel)
├── model_BO1.pkl            # Modèles ML
├── model_BO2.pkl
├── model_BO3.pkl
├── model_BO4.pkl
├── scaler_BO1.pkl           # Scalers
├── scaler_BO2.pkl
├── scaler_BO3.pkl
├── scaler_BO4.pkl
├── features_BO1.pkl         # Listes de features
├── features_BO2.pkl
├── features_BO3.pkl
├── features_BO4.pkl
├── top_5_features_BO3.pkl
├── cluster_info_BO4.csv     # Données de clustering
├── heart_disease.csv        # Dataset (optionnel, si nécessaire)
└── README.md                # Documentation
```

## ⚙️ Configuration Optionnelle

### Créer un fichier `.streamlit/config.toml` (optionnel)

Créez un dossier `.streamlit` et un fichier `config.toml` pour personnaliser la configuration :

```toml
[theme]
primaryColor = "#14c7dd"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#15171a"
textColor = "#ffffff"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
```

### Créer un fichier `.gitignore` (recommandé)

Créez un fichier `.gitignore` pour exclure les fichiers inutiles :

```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv
*.log
.DS_Store
*.ipynb_checkpoints
```

## 🔄 Mises à Jour

Pour mettre à jour votre application déployée :

1. **Modifier votre code localement**
2. **Commit et push vers GitHub** :
   ```bash
   git add .
   git commit -m "Description des modifications"
   git push
   ```
3. **Streamlit Cloud redéploiera automatiquement** votre application

## 🐛 Résolution de Problèmes

### Erreur : "Module not found"
- Vérifiez que toutes les dépendances sont dans `requirements.txt`
- Vérifiez que les versions sont compatibles

### Erreur : "File not found" (fichiers .pkl ou .csv)
- Vérifiez que tous les fichiers nécessaires sont dans le repository
- Vérifiez les chemins dans `app.py` (utilisez des chemins relatifs)

### L'application ne se charge pas
- Vérifiez les logs dans Streamlit Cloud (onglet "Manage app" → "Logs")
- Vérifiez que `app.py` est le fichier principal et qu'il n'y a pas d'erreurs de syntaxe

### Les modèles ne se chargent pas
- Vérifiez que tous les fichiers `.pkl` sont bien commités et pushés
- Vérifiez que les chemins dans `app.py` sont corrects (ex: `'model_BO1.pkl'` et non `'./model_BO1.pkl'`)

## 📊 Limites de Streamlit Cloud (Gratuit)

- **CPU** : 1 core
- **RAM** : 1 GB
- **Stockage** : 1 GB
- **Bande passante** : Illimitée
- **Applications publiques uniquement**

## 🔒 Sécurité

- Ne commitez **jamais** de données sensibles (mots de passe, clés API)
- Utilisez des variables d'environnement pour les secrets (via Streamlit Cloud → Settings → Secrets)

## 📚 Ressources

- [Documentation Streamlit Cloud](https://docs.streamlit.io/streamlit-community-cloud)
- [Forum Streamlit](https://discuss.streamlit.io/)
- [GitHub Streamlit](https://github.com/streamlit/streamlit)

## ✅ Checklist de Déploiement

- [ ] Compte GitHub créé
- [ ] Compte Streamlit Cloud créé
- [ ] Repository GitHub créé
- [ ] Tous les fichiers nécessaires dans le projet
- [ ] `requirements.txt` à jour
- [ ] Code testé localement
- [ ] Projet initialisé avec Git
- [ ] Code pushé sur GitHub
- [ ] Application déployée sur Streamlit Cloud
- [ ] Application testée en ligne

---

**Bon déploiement ! 🚀**

