"""
Application Streamlit pour la démonstration des modèles de prédiction de risque cardiaque
Design moderne style Skydash Dashboard - Thème clair
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(
    page_title="Système de Prédiction du Risque Cardiaque",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé style Skydash - Thème clair
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" integrity="sha512-iecdLmaskl7CVkqkXNQ/ZH/XLlvWZOJyj7Yy7tcenmpD1ypASozpmT/E0iPtmFIB46ZmdtAc9eNBvH0H/ZpiBw==" crossorigin="anonymous" referrerpolicy="no-referrer" />
<style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
    /* Reset */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    /* Assurer que le HTML est rendu correctement */
    .stMarkdown {
        white-space: normal !important;
    }
    
    /* Forcer le rendu HTML dans les markdown */
    [data-testid="stMarkdownContainer"] {
        white-space: normal !important;
    }
    
    /* S'assurer que le HTML est bien rendu et non affiché comme texte brut */
    [data-testid="stMarkdownContainer"] > div {
        white-space: normal !important;
    }
    
    /* Ne pas traiter le HTML comme du code */
    [data-testid="stMarkdownContainer"] pre code {
        display: none !important;
    }
    
    /* Thème sombre Skydash - Fond général */
    .main {
        background: #212121;
        padding: 0;
    }
    
    /* Assurer que le texte principal est clair et lisible */
    .main .block-container,
    .main [data-testid="stAppViewContainer"],
    .main [data-testid="stAppViewBlockContainer"] {
        background: #212121;
        color: #ffffff !important;
    }
    
    .main p, .main span, .main div, .main li, .main td, .main th {
        color: #ffffff !important;
    }
    
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {
        color: #cfcfd0 !important;
    }
    
    /* Forcer les titres h1 et h2 en gris clair */
    h1, h2 {
        color: #cfcfd0 !important;
        font-weight: 700 !important;
    }
    
    /* Forcer TOUS les paragraphes en blanc - Règles très spécifiques */
    p, p *, p span, p strong, p b, p em {
        color: #ffffff !important;
    }
    
    /* Paragraphes dans tous les conteneurs */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] p *,
    [data-testid="stAppViewBlockContainer"] p,
    [data-testid="stAppViewBlockContainer"] p *,
    .block-container p,
    .block-container p *,
    .main p,
    .main p * {
        color: #ffffff !important;
    }
    
    .main strong, .main b {
        color: #ffffff !important;
    }
    
    /* Forcer tous les textes en blanc */
    span, div, li, td, th, label {
        color: #ffffff !important;
    }
    
    /* Sidebar style Skydash - Bleu clair */
    [data-testid="stSidebar"] {
        background: #0c7885;
        border-right: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Bouton de toggle de la sidebar - Visible et stylisé */
    button[data-testid="baseButton-header"] {
        background: #0c7885 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
        position: fixed !important;
        top: 0.5rem !important;
        left: 0.5rem !important;
        z-index: 999 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    button[data-testid="baseButton-header"]:hover {
        background: #2d8fd4 !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }
    
    button[data-testid="baseButton-header"] svg {
        fill: #ffffff !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    /* Alternative: Bouton hamburger personnalisé */
    .sidebar-toggle-btn {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 999;
        background: #0c7885;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
        padding: 0.75rem;
        border-radius: 6px;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    
    .sidebar-toggle-btn:hover {
        background: #2d8fd4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    /* Texte blanc uniquement pour la sidebar */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 0;
    }
    
    /* Header style Skydash - Thème sombre */
    .dashboard-header {
        background: #15171a;
        padding: 1.5rem 2rem;
        border-bottom: 1px solid #334155;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .dashboard-header h1 {
        color: #cfcfd0 !important;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    
    .dashboard-header p {
        color: #ffffff !important;
        margin: 0.5rem 0 0 0;
        font-size: 0.95rem;
    }
    
    .dashboard-header i {
        color: #ffffff !important;
    }
    
    /* Cartes de métriques style Bootstrap Alert */
    .metric-card {
        border-radius: 0.375rem;
        padding: 1rem 1.25rem;
        margin-bottom: 1.5rem;
        height: 100%;
        border: 1px solid transparent;
        border-left: 4px solid;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    .metric-card.primary { 
        border-left-color: #0d6efd;
        background-color: #15171a;
        border-color: #334155;
        color: #ffffff !important;
    }
    
    .metric-card.success { 
        border-left-color: #198754;
        background-color: #15171a;
        border-color: #334155;
        color: #ffffff !important;
    }
    
    .metric-card.warning { 
        border-left-color: #ffc107;
        background-color: #15171a;
        border-color: #334155;
        color: #ffffff !important;
    }
    
    .metric-card.danger { 
        border-left-color: #dc3545;
        background-color: #15171a;
        border-color: #334155;
        color: #ffffff !important;
    }
    
    .metric-card.info { 
        border-left-color: #0dcaf0;
        background-color: #15171a;
        border-color: #334155;
        color: #ffffff !important;
    }
    
    /* Tous les textes dans les cartes métriques en blanc */
    .metric-card * {
        color: #ffffff !important;
    }
    
    .metric-icon {
        width: 48px;
        height: 48px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Icônes métriques avec couleurs Bootstrap */
    .metric-icon.primary { 
        background: rgba(13, 110, 253, 0.1); 
        color: #0d6efd !important; 
        border: 1px solid rgba(13, 110, 253, 0.3); 
    }
    .metric-icon.success { 
        background: rgba(25, 135, 84, 0.1); 
        color: #198754 !important; 
        border: 1px solid rgba(25, 135, 84, 0.3); 
    }
    .metric-icon.warning { 
        background: rgba(255, 193, 7, 0.1); 
        color: #ffc107 !important; 
        border: 1px solid rgba(255, 193, 7, 0.3); 
    }
    .metric-icon.danger { 
        background: rgba(220, 53, 69, 0.1); 
        color: #dc3545 !important; 
        border: 1px solid rgba(220, 53, 69, 0.3); 
    }
    .metric-icon.info { 
        background: rgba(13, 202, 240, 0.1); 
        color: #0dcaf0 !important; 
        border: 1px solid rgba(13, 202, 240, 0.3); 
    }
    
    /* Icônes dans les cartes métriques selon leur type */
    .metric-card.primary i,
    .metric-card.primary .metric-icon {
        color: #0d6efd !important;
    }
    
    .metric-card.success i,
    .metric-card.success .metric-icon {
        color: #198754 !important;
    }
    
    .metric-card.warning i,
    .metric-card.warning .metric-icon {
        color: #ffc107 !important;
    }
    
    .metric-card.danger i,
    .metric-card.danger .metric-icon {
        color: #dc3545 !important;
    }
    
    .metric-card.info i,
    .metric-card.info .metric-icon {
        color: #0dcaf0 !important;
    }
    
    /* Icônes dans les alertes Bootstrap */
    .alert-primary i {
        color: #084298 !important;
    }
    
    .alert-success i {
        color: #0f5132 !important;
    }
    
    .alert-warning i {
        color: #664d03 !important;
    }
    
    .alert-danger i {
        color: #842029 !important;
    }
    
    .alert-info i {
        color: #055160 !important;
    }
    
    /* Icônes dans les badges Bootstrap */
    .badge-primary i {
        color: #0d6efd !important;
    }
    
    .badge-success i {
        color: #198754 !important;
    }
    
    .badge-warning i {
        color: #ffc107 !important;
    }
    
    .badge-danger i {
        color: #dc3545 !important;
    }
    
    .badge-info i {
        color: #0dcaf0 !important;
    }
    
    /* Icônes dans les cartes de contenu selon leur type */
    .content-card.primary i {
        color: #0d6efd !important;
    }
    
    .content-card.success i {
        color: #198754 !important;
    }
    
    .content-card.warning i {
        color: #ffc107 !important;
    }
    
    .content-card.danger i {
        color: #dc3545 !important;
    }
    
    .content-card.info i {
        color: #0dcaf0 !important;
    }
    
    /* Icônes par défaut en blanc (pour les autres cas) */
    i, .fas, .fa, [class*="fa-"], [class^="fa-"], [class*=" fas "], [class*=" fa "] {
        color: #ffffff !important;
    }
    
    /* Icônes dans les titres - blanc par défaut */
    h1 i, h2 i, h3 i, h4 i, h5 i, h6 i,
    .card-title i, .dashboard-header i {
        color: #ffffff !important;
    }
    
    /* Forcer toutes les icônes Font Awesome par défaut */
    .fa, .fas, .far, .fal, .fab {
        color: #ffffff !important;
    }
    
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #ffffff !important;
    }
    
    .metric-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #ffffff !important;
    }
    
    /* Cartes de contenu style Skydash - Thème sombre */
    .content-card {
        background: #15171a;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        border: 1px solid #334155;
        border-left: 4px solid #5d87ff;
    }
    
    .content-card.primary {
        border-left-color: #5d87ff;
        background: #15171a;
    }
    
    .content-card.success {
        border-left-color: #13deb9;
        background: #15171a;
    }
    
    .content-card.warning {
        border-left-color: #ffae1f;
        background: #15171a;
    }
    
    .content-card.danger {
        border-left-color: #fa896b;
        background: #15171a;
    }
    
    .content-card.info {
        border-left-color: #49beff;
        background: #15171a;
    }
    
    /* Texte dans les cartes de contenu - Appliquer seulement si pas de style inline */
    .content-card h1:not([style*="color:"]), 
    .content-card h2:not([style*="color:"]), 
    .content-card h3:not([style*="color:"]), 
    .content-card h4:not([style*="color:"]) {
        color: #cfcfd0 !important;
        font-weight: 600;
    }
    
    .content-card h1, .content-card h2, .content-card h3, .content-card h4 {
        font-weight: 600;
    }
    
    .content-card p, .content-card span, .content-card li {
        color: #ffffff !important;
    }
    
    .content-card strong, .content-card b {
        color: #ffffff !important;
    }
    
    .card-header {
        border-bottom: 1px solid #334155;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #cfcfd0 !important;
        margin: 0;
    }
    
    .card-title i {
        color: #ffffff !important;
    }
    
    /* Boutons style Skydash */
    .stButton>button {
        background: #14c7dd;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.625rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
        width: 100%;
        box-shadow: 0 2px 4px rgba(20, 199, 221, 0.2);
    }
    
    .stButton>button:hover {
        background: #0fa8c0;
        box-shadow: 0 4px 12px rgba(20, 199, 221, 0.3);
        transform: translateY(-1px);
    }
    
    /* Inputs style Skydash - Thème sombre */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 6px;
        border: 1px solid #475569;
        padding: 0.5rem;
        font-size: 0.95rem;
        background: #15171a;
        color: #ffffff;
    }
    
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #5d87ff;
        box-shadow: 0 0 0 3px rgba(93, 135, 255, 0.2);
        background: #15171a;
        color: #ffffff;
    }
    
    .stNumberInput label,
    .stSelectbox label {
        color: #ffffff !important;
    }
    
    /* Labels - Thème sombre */
    label {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Titres - Couleur gris clair pour tous les titres */
    h1 {
        color: #cfcfd0 !important;
        font-weight: 700 !important;
    }
    
    h2 {
        color: #cfcfd0 !important;
        font-weight: 700 !important;
        font-size: 1.5rem;
    }
    
    /* Titres h3 - Appliquer la couleur seulement si pas de style inline */
    h3:not([style*="color:"]) {
        color: #cfcfd0 !important;
    }
    
    h3 {
        font-weight: 600;
        font-size: 1.125rem;
    }
    
    /* Permettre aux couleurs inline des niveaux de risque de s'afficher */
    /* Les styles inline ont naturellement la priorité, mais on s'assure qu'ils ne sont pas écrasés */
    h3[style*="color:"],
    div[style*="color:"] {
        /* Laisser les styles inline fonctionner - ne pas forcer de couleur */
    }
    
    h4, h5, h6 {
        color: #cfcfd0 !important;
        font-weight: 600;
    }
    
    /* Paragraphes et texte - Blanc - Règles très fortes */
    p, p *, p span, p strong, p b, p em, p i {
        color: #ffffff !important;
    }
    
    /* Paragraphes dans tous les contextes HTML */
    body p, html p, div p, section p, article p, main p {
        color: #ffffff !important;
    }
    
    /* Paragraphes Streamlit spécifiques */
    .stMarkdown p,
    .stMarkdown p *,
    .element-container p,
    .element-container p *,
    .block-container p,
    .block-container p * {
        color: #ffffff !important;
    }
    
    /* Listes */
    ul, ol, li, li *, li span, li strong {
        color: #ffffff !important;
    }
    
    /* Texte en gras */
    strong, b, strong *, b * {
        color: #ffffff !important;
    }
    
    /* Tous les éléments de texte */
    span, div, label, td, th {
        color: #ffffff !important;
    }
    
    /* Assurer que le contenu principal a du texte blanc */
    [data-testid="stAppViewContainer"] p,
    [data-testid="stAppViewContainer"] p *,
    [data-testid="stAppViewContainer"] span,
    [data-testid="stAppViewContainer"] div,
    [data-testid="stAppViewContainer"] li {
        color: #ffffff !important;
    }
    
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4,
    [data-testid="stAppViewContainer"] h5,
    [data-testid="stAppViewContainer"] h6 {
        color: #cfcfd0 !important;
    }
    
    /* Alertes style Skydash */
    .alert {
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    
    .alert-info {
        background: rgba(73, 190, 255, 0.1);
        border-left-color: #49beff;
        color: #ffffff;
    }
    
    .alert-success {
        background: rgba(19, 222, 185, 0.1);
        border-left-color: #13deb9;
        color: #ffffff;
    }
    
    .alert-warning {
        background: rgba(255, 174, 31, 0.1);
        border-left-color: #ffae1f;
        color: #ffffff;
    }
    
    .alert-danger {
        background: rgba(250, 137, 107, 0.1);
        border-left-color: #fa896b;
        color: #ffffff;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.375rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-success { background: rgba(19, 222, 185, 0.2); color: #13deb9; border: 1px solid rgba(19, 222, 185, 0.3); }
    .badge-warning { background: rgba(255, 174, 31, 0.2); color: #ffae1f; border: 1px solid rgba(255, 174, 31, 0.3); }
    .badge-danger { background: rgba(250, 137, 107, 0.2); color: #fa896b; border: 1px solid rgba(250, 137, 107, 0.3); }
    .badge-info { background: rgba(73, 190, 255, 0.2); color: #49beff; border: 1px solid rgba(73, 190, 255, 0.3); }
    .badge-primary { background: rgba(93, 135, 255, 0.2); color: #5d87ff; border: 1px solid rgba(93, 135, 255, 0.3); }
    
    /* Navigation sidebar */
    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        transition: all 0.3s;
        margin: 0.25rem 0;
    }
    
    [data-testid="stSidebar"] label:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        color: white !important;
    }
    
    /* Sidebar header */
    .sidebar-header {
        padding: 1.5rem 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.3);
        margin-bottom: 1rem;
    }
    
    .sidebar-header h2 {
        color: #ffffff !important;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .sidebar-header p {
        color: rgba(255, 255, 255, 0.95) !important;
        font-size: 0.875rem;
        margin: 0.25rem 0 0 0;
    }
    
    .sidebar-header i {
        color: #ffffff !important;
    }
    
    /* Section titles */
    .section-title {
        color: #2c3e50;
        font-size: 1rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    
    /* Hide Streamlit branding et menu Git */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Masquer l'icône Git et le lien vers le code source */
    [data-testid="stHeader"] a[href*="github"] {
        display: none !important;
    }
    
    /* Masquer le menu déroulant avec les options */
    [data-testid="stHeader"] [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Masquer le bouton "Deploy" si présent */
    .stDeployButton {
        display: none !important;
    }
    
    /* Masquer tous les liens dans le header sauf le bouton sidebar */
    [data-testid="stHeader"] a:not([data-testid="baseButton-header"]) {
        display: none !important;
    }
    
    /* Masquer l'icône GitHub dans le header (sélecteurs alternatifs) */
    [data-testid="stHeader"] [href*="github"],
    [data-testid="stHeader"] [href*="GitHub"],
    [data-testid="stHeader"] svg[viewBox*="16"]:has(+ a[href*="github"]),
    [data-testid="stHeader"] a[aria-label*="GitHub"],
    [data-testid="stHeader"] a[aria-label*="github"],
    [data-testid="stHeader"] a[title*="GitHub"],
    [data-testid="stHeader"] a[title*="github"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Masquer le menu hamburger avec les options (About, Settings, etc.) */
    [data-testid="stHeader"] [data-testid="stToolbar"] button,
    [data-testid="stHeader"] [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Masquer les éléments du menu déroulant */
    [data-testid="stHeader"] [role="menu"],
    [data-testid="stHeader"] [role="menuitem"] {
        display: none !important;
    }
    
    /* Ne pas cacher le header pour garder le bouton de toggle */
    /* header {visibility: hidden;} */
    
    /* Assurer que le bouton de toggle est visible */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    [data-testid="stHeader"] button {
        background: #0c7885 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
    }
    
    [data-testid="stHeader"] button:hover {
        background: #2d8fd4 !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }
    
    [data-testid="stHeader"] button svg {
        fill: #ffffff !important;
    }
    
    /* Stats grid */
    .stats-item {
        text-align: center;
        padding: 1.5rem;
        background: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #5d87ff;
        margin: 0.5rem 0;
    }
    
    .stats-label {
        color: #6c757d;
        font-size: 0.875rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Chargement des modèles et scalers
@st.cache_resource
def load_models():
    """Charge tous les modèles et scalers"""
    try:
        models = {
            'BO1': {
                'model': joblib.load('model_BO1.pkl'),
                'scaler': joblib.load('scaler_BO1.pkl'),
                'features': joblib.load('features_BO1.pkl')
            },
            'BO2': {
                'model': joblib.load('model_BO2.pkl'),
                'scaler': joblib.load('scaler_BO2.pkl'),
                'features': joblib.load('features_BO2.pkl')
            },
            'BO3': {
                'model': joblib.load('model_BO3.pkl'),
                'scaler': joblib.load('scaler_BO3.pkl'),
                'features': joblib.load('features_BO3.pkl'),
                'top_features': joblib.load('top_5_features_BO3.pkl')
            },
            'BO4': {
                'model': joblib.load('model_BO4.pkl'),
                'scaler': joblib.load('scaler_BO4.pkl'),
                'features': joblib.load('features_BO4.pkl'),
                'cluster_info': pd.read_csv('cluster_info_BO4.csv')
            }
        }
        return models
    except FileNotFoundError as e:
        st.error(f"Erreur: Fichier modèle non trouvé. Veuillez d'abord exécuter train_models.py")
        st.stop()

# Sidebar avec style Skydash
st.sidebar.markdown("""
<div class='sidebar-header'>
    <h2><i class="fas fa-heartbeat" style="color: white !important; margin-right: 0.5rem;"></i> CardioAI</h2>
    <p style="color: rgba(255,255,255,0.9) !important;">Dashboard de Prédiction</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Classification Binaire", "Score Continu", 
     "Classification Multi-classe", "Clustering"],
    label_visibility="collapsed"
)

# ========== PAGE DASHBOARD ==========
if page == "Dashboard":
    st.markdown("""
    <div class='dashboard-header'>
        <h1><i class="fas fa-tachometer-alt"></i> Tableau de Bord</h1>
        <p>Système de Prédiction du Risque Cardiaque - Intelligence Artificielle pour la Prévention</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistiques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card primary'>
            <div class='metric-icon primary'>
                <i class="fas fa-brain"></i>
            </div>
            <div class='metric-value'>4</div>
            <div class='metric-label'>Modèles IA</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card success'>
            <div class='metric-icon success'>
                <i class="fas fa-users"></i>
            </div>
            <div class='metric-value'>4,238</div>
            <div class='metric-label'>Patients Analysés</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card warning'>
            <div class='metric-icon warning'>
                <i class="fas fa-chart-line"></i>
            </div>
            <div class='metric-value'>16</div>
            <div class='metric-label'>Variables Cliniques</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='metric-card danger'>
            <div class='metric-icon danger'>
                <i class="fas fa-bullseye"></i>
            </div>
            <div class='metric-value'>95%</div>
            <div class='metric-label'>Précision Moyenne</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Contenu principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='content-card primary'>
            <div class='card-header'>
                <h2 class='card-title'><i class="fas fa-info-circle"></i> Vue d'Ensemble du Projet</h2>
            </div>
            <h3 style='color: #cfcfd0; margin-top: 1rem;'><i class="fas fa-question-circle" style="color: #ffffff !important;"></i> Problématique</h3>
            <p style='line-height: 1.8; color: #ffffff; margin-bottom: 1.5rem;'>
            Les maladies cardiovasculaires représentent l'une des principales causes de mortalité dans le monde. 
            L'identification précoce des patients à risque permet une intervention médicale rapide et efficace, 
            réduisant ainsi la morbidité et la mortalité associées.
            </p>
            
            <h3 style='color: #cfcfd0; margin-top: 1.5rem;'><i class="fas fa-lightbulb" style="color: #ffffff !important;"></i> Solution Proposée</h3>
            <p style='line-height: 1.8; color: #ffffff; margin-bottom: 1rem;'>
            Cette application propose une solution complète basée sur l'intelligence artificielle pour :
            </p>
            <ul style='line-height: 2.5; color: #ffffff; padding-left: 1.5rem;'>
                <li><strong style='color: #ffffff;'>Identifier précocement</strong> les patients asymptomatiques à haut risque</li>
                <li><strong style='color: #ffffff;'>Évaluer le risque</strong> avec un score continu personnalisé (0-100)</li>
                <li><strong  style='color: #d4edda;' >Classer les patients</strong> selon leur niveau de risque (faible/moyen/élevé)</li>
                <li><strong style='color: #ffffff;'>Grouper les patients</strong> similaires pour une meilleure compréhension des profils</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-card success'>
            <div class='card-header'>
                <h2 class='card-title'><i class="fas fa-database"></i> Source des Données</h2>
            </div>
            <div style='margin-top: 1rem;'>
                <p style='margin-bottom: 0.5rem; color: #ffffff; font-weight: 600;'><i class="fas fa-file-alt" style="color: #ffffff !important;"></i> Dataset :</p>
                <p style='color: #ffffff; margin-bottom: 1.5rem;'>Heart Disease Dataset</p>
                
                <p style='margin-bottom: 0.5rem; color: #ffffff; font-weight: 600;'><i class="fas fa-users" style="color: #ffffff !important;"></i> Nombre de patients :</p>
                <p style='color: #ffffff; margin-bottom: 1.5rem;'>4,238 patients</p>
                
                <p style='margin-bottom: 0.5rem; color: #ffffff; font-weight: 600;'><i class="fas fa-list" style="color: #ffffff !important;"></i> Variables :</p>
                <p style='color: #ffffff; margin-bottom: 1.5rem;'>16 caractéristiques</p>
                
                <p style='margin-bottom: 0.5rem; color: #ffffff; font-weight: 600;'><i class="fas fa-key" style="color: #ffffff !important;"></i> Variables clés :</p>
                <ul style='color: #ffffff; padding-left: 1.5rem; line-height: 2;'>
                    <li>Âge</li>
                    <li>Pression artérielle</li>
                    <li>Cholestérol</li>
                    <li>BMI</li>
                    <li>Tabagisme</li>
                    <li>Hypertension</li>
                    <li>Diabète</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Objectifs métier
    st.markdown("""
    <div class='content-card warning'>
        <div class='card-header'>
            <h2 class='card-title'><i class="fas fa-bullseye"></i> Objectifs Métier</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='content-card' style='border-left: 4px solid #5d87ff;'>
            <h3 style='color: #cfcfd0; margin-top: 0;'><i class="fas fa-check-circle" style="color: #ffffff !important;"></i> Classification Binaire</h3>
            <p style='color: #ffffff; line-height: 1.8;'>
            Identifier les patients asymptomatiques à haut risque à partir de facteurs tels que le tabagisme, 
            l'hypertension et le diabète.
            </p>
            <div style='margin-top: 1rem;'>
                <span class='badge badge-primary'><i class="fas fa-robot"></i> XGBoost Classifier</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='content-card' style='border-left: 4px solid #13deb9;'>
            <h3 style='color: #cfcfd0; margin-top: 0;'><i class="fas fa-chart-area" style="color: #ffffff !important;"></i> Score Continu</h3>
            <p style='color: #ffffff; line-height: 1.8;'>
            Construire un modèle de régression robuste pour prédire un score de risque continu 
            personnalisé (0-100).
            </p>
            <div style='margin-top: 1rem;'>
                <span class='badge badge-success'><i class="fas fa-robot"></i> XGBoost Regressor</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='content-card' style='border-left: 4px solid #ffae1f;'>
            <h3 style='color: #cfcfd0; margin-top: 0;'><i class="fas fa-layer-group" style="color: #ffffff !important;"></i> Classification Multi-classe</h3>
            <p style='color: #ffffff; line-height: 1.8;'>
            Identifier les cinq facteurs les plus prédictifs et classer les patients selon leur niveau 
            de risque (faible/moyen/élevé).
            </p>
            <div style='margin-top: 1rem;'>
                <span class='badge badge-warning'><i class="fas fa-tree"></i> Random Forest</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='content-card' style='border-left: 4px solid #fa896b;'>
            <h3 style='color: #cfcfd0; margin-top: 0;'><i class="fas fa-project-diagram" style="color: #ffffff !important;"></i> Clustering</h3>
            <p style='color: #ffffff; line-height: 1.8;'>
            Former des clusters cohérents basés sur les données comportementales et cliniques 
            pour identifier des groupes de patients similaires.
            </p>
            <div style='margin-top: 1rem;'>
                <span class='badge badge-danger'><i class="fas fa-sitemap"></i> K-Means</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Pipeline
    st.markdown("""
    <div class='content-card info'>
        <div class='card-header'>
            <h2 class='card-title'><i class="fas fa-cogs"></i> Pipeline de Traitement</h2>
        </div>
        <div style='display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-top: 1.5rem;'>
            <div style='flex: 1; min-width: 150px; text-align: center; padding: 1.5rem; background: #334155; border-radius: 8px; border: 1px solid #475569;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem; color: #ffffff;'><i class="fas fa-broom" style="color: #ffffff !important;"></i></div>
                <div style='font-weight: 600; color: #ffffff;'>Nettoyage</div>
            </div>
            <div style='flex: 1; min-width: 150px; text-align: center; padding: 1.5rem; background: #334155; border-radius: 8px; border: 1px solid #475569;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem; color: #13deb9;'><i class="fas fa-tools"></i></div>
                <div style='font-weight: 600; color: #ffffff;'>Feature Engineering</div>
            </div>
            <div style='flex: 1; min-width: 150px; text-align: center; padding: 1.5rem; background: #334155; border-radius: 8px; border: 1px solid #475569;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem; color: #ffae1f;'><i class="fas fa-sliders-h"></i></div>
                <div style='font-weight: 600; color: #ffffff;'>Normalisation</div>
            </div>
            <div style='flex: 1; min-width: 150px; text-align: center; padding: 1.5rem; background: #334155; border-radius: 8px; border: 1px solid #475569;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem; color: #fa896b;'><i class="fas fa-filter"></i></div>
                <div style='font-weight: 600; color: #ffffff;'>Sélection Features</div>
            </div>
            <div style='flex: 1; min-width: 150px; text-align: center; padding: 1.5rem; background: #334155; border-radius: 8px; border: 1px solid #475569;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem; color: #49beff;'><i class="fas fa-graduation-cap"></i></div>
                <div style='font-weight: 600; color: #ffffff;'>Entraînement</div>
            </div>
            <div style='flex: 1; min-width: 150px; text-align: center; padding: 1.5rem; background: #334155; border-radius: 8px; border: 1px solid #475569;'>
                <div style='font-size: 2rem; margin-bottom: 0.5rem; color: #ffffff;'><i class="fas fa-chart-bar"></i></div>
                <div style='font-weight: 600; color: #ffffff;'>Évaluation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== BO1 : CLASSIFICATION BINAIRE ==========
elif page == "Classification Binaire":
    st.markdown("""
    <div class='dashboard-header'>
        <h1><i class="fas fa-check-double"></i> Classification Binaire</h1>
        <p>BO1 : Identification des Patients à Haut Risque - Prédiction du risque cardiaque (Haut risque / Faible risque)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='alert alert-info'>
        <strong><i class="fas fa-info-circle"></i> Objectif :</strong> Développer un modèle capable de prédire si un patient asymptomatique appartient 
        à la catégorie "haut risque", à partir de facteurs tels que le tabagisme, l'hypertension et le diabète.
        <br><strong><i class="fas fa-robot"></i> Modèle :</strong> XGBoost Classifier
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    features = models['BO1']['features']
    
    st.markdown("""
    <div class='content-card primary'>
        <div class='card-header'>
            <h2 class='card-title'><i class="fas fa-edit"></i> Formulaire de Saisie</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <h4 style='color: #cfcfd0; margin-bottom: 1rem;'><i class="fas fa-user"></i> Informations Démographiques</h4>
        """, unsafe_allow_html=True)
        age = st.number_input("Âge", min_value=18, max_value=100, value=50, step=1)
        sysBP = st.number_input("Pression Systolique (mmHg)", min_value=80.0, max_value=250.0, 
                               value=120.0, step=0.1)
        diaBP = st.number_input("Pression Diastolique (mmHg)", min_value=40.0, max_value=150.0, 
                               value=80.0, step=0.1)
        totChol = st.number_input("Cholestérol Total (mg/dL)", min_value=100.0, max_value=400.0, 
                                 value=200.0, step=1.0)
    
    with col2:
        st.markdown("""
        <h4 style='color: #cfcfd0; margin-bottom: 1rem;'><i class="fas fa-heartbeat"></i> Indicateurs Physiques</h4>
        """, unsafe_allow_html=True)
        BMI = st.number_input("IMC (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
        heartRate = st.number_input("Fréquence Cardiaque (bpm)", min_value=40, max_value=120, 
                                   value=75, step=1)
        glucose = st.number_input("Glycémie (mg/dL)", min_value=50.0, max_value=300.0, 
                                 value=85.0, step=1.0)
        currentSmoker = st.selectbox("Fumeur Actuel", [0, 1], 
                                    format_func=lambda x: "Non" if x == 0 else "Oui")
    
    with col3:
        st.markdown("""
        <h4 style='color: #cfcfd0; margin-bottom: 1rem;'><i class="fas fa-exclamation-triangle"></i> Facteurs de Risque</h4>
        """, unsafe_allow_html=True)
        prevalentHyp = st.selectbox("Hypertension", [0, 1], 
                                   format_func=lambda x: "Non" if x == 0 else "Oui")
        diabetes = st.selectbox("Diabète", [0, 1], 
                               format_func=lambda x: "Non" if x == 0 else "Oui")
    
    if st.button("Analyser le Risque", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'age': [age], 'sysBP': [sysBP], 'diaBP': [diaBP], 'totChol': [totChol],
            'BMI': [BMI], 'heartRate': [heartRate], 'glucose': [glucose],
            'currentSmoker': [currentSmoker], 'prevalentHyp': [prevalentHyp], 'diabetes': [diabetes]
        })
        
        input_scaled = models['BO1']['scaler'].transform(input_data)
        prediction = models['BO1']['model'].predict(input_scaled)[0]
        probability = models['BO1']['model'].predict_proba(input_scaled)[0]
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class='content-card' style='border-left: 4px solid #fa896b; background: #15171a;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 3rem; color: #fa896b; margin-bottom: 1rem;'><i class="fas fa-exclamation-triangle"></i></div>
                        <h3 style='color: #fa896b; margin: 1rem 0;'>RISQUE ÉLEVÉ</h3>
                        <div style='font-size: 2.5rem; font-weight: 700; color: #fa896b;'>{probability[1]*100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='content-card' style='border-left: 4px solid #13deb9; background: #15171a;'>
                    <div style='text-align: center;'>
                        <div style='font-size: 3rem; color: #13deb9; margin-bottom: 1rem;'><i class="fas fa-check-circle"></i></div>
                        <h3 style='color: #13deb9; margin: 1rem 0;'>RISQUE FAIBLE</h3>
                        <div style='font-size: 2.5rem; font-weight: 700; color: #13deb9;'>{probability[0]*100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure()
            colors = ['#13deb9', '#fa896b'] if prediction == 0 else ['#fa896b', '#13deb9']
            fig.add_trace(go.Bar(
                x=['Risque Faible', 'Risque Élevé'],
                y=[probability[0]*100, probability[1]*100],
                marker=dict(color=colors, line=dict(color='white', width=2)),
                text=[f'{probability[0]*100:.1f}%', f'{probability[1]*100:.1f}%'],
                textposition='auto',
                textfont=dict(size=16, color='white', family='Arial Black')
            ))
            fig.update_layout(
                title=dict(text="Probabilités de Prédiction", font=dict(size=20, color='#2c3e50'), x=0.5),
                yaxis=dict(title="Probabilité (%)", range=[0, 100]),
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='alert alert-warning'>
            <strong><i class="fas fa-lightbulb"></i> Recommandation :</strong> Consultez un professionnel de santé pour une évaluation complète et personnalisée.
        </div>
        """, unsafe_allow_html=True)

# ========== BO2 : SCORE CONTINU ==========
elif page == "Score Continu":
    st.markdown("""
    <div class='dashboard-header'>
        <h1><i class="fas fa-chart-area"></i> Score de Risque Comportemental Continu</h1>
        <p>BO2 : Prédiction d'un score de risque continu (0-100)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='alert alert-info'>
        <strong><i class="fas fa-info-circle"></i> Objectif :</strong> Construire un modèle de régression robuste pour prédire un score de risque continu 
        à partir des variables spécifiées.
        <br><strong><i class="fas fa-robot"></i> Modèle :</strong> XGBoost Regressor | <strong><i class="fas fa-ruler"></i> Score :</strong> 0-100 (0 = risque minimal, 100 = risque maximal)
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    
    st.markdown("""
    <div class='content-card success'>
        <div class='card-header'>
            <h2 class='card-title'><i class="fas fa-edit"></i> Formulaire de Saisie</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=50, step=1, key="bo2_age")
        sysBP = st.number_input("Pression Systolique (mmHg)", min_value=80.0, max_value=250.0, 
                               value=120.0, step=0.1, key="bo2_sysBP")
        totChol = st.number_input("Cholestérol Total (mg/dL)", min_value=100.0, max_value=400.0, 
                                 value=200.0, step=1.0, key="bo2_totChol")
    
    with col2:
        BMI = st.number_input("IMC (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, key="bo2_BMI")
        glucose = st.number_input("Glycémie (mg/dL)", min_value=50.0, max_value=300.0, 
                                 value=85.0, step=1.0, key="bo2_glucose")
    
    if st.button("Calculer le Score de Risque", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'age': [age], 'sysBP': [sysBP], 'totChol': [totChol],
            'BMI': [BMI], 'glucose': [glucose]
        })
        
        input_scaled = models['BO2']['scaler'].transform(input_data)
        score = models['BO2']['model'].predict(input_scaled)[0]
        score = max(0, min(100, score))
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if score < 33:
                risk_level = "FAIBLE"
                risk_color = "#13deb9"
                risk_bg = "#15171a"
                risk_icon = "check-circle"
            elif score < 66:
                risk_level = "MOYEN"
                risk_color = "#ffae1f"
                risk_bg = "#15171a"
                risk_icon = "exclamation-circle"
            else:
                risk_level = "ÉLEVÉ"
                risk_color = "#fa896b"
                risk_bg = "#15171a"
                risk_icon = "exclamation-triangle"
            
            st.markdown(f"""
            <div class='content-card' style='border-left: 4px solid {risk_color}; background: {risk_bg};'>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem; color: {risk_color}; margin-bottom: 1rem;'><i class="fas fa-{risk_icon}"></i></div>
                    <h3 style='color: {risk_color}; margin: 1rem 0;'>Niveau de Risque</h3>
                    <div style='font-size: 2.5rem; font-weight: 700; color: {risk_color};'>{risk_level}</div>
                    <div style='font-size: 3rem; font-weight: 700; color: {risk_color}; margin-top: 1rem;'>{score:.1f}/100</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Score de Risque Cardiaque", 'font': {'size': 24}},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#5d87ff"},
                    'steps': [
                        {'range': [0, 33], 'color': "#13deb9"},
                        {'range': [33, 66], 'color': "#ffae1f"},
                        {'range': [66, 100], 'color': "#fa896b"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='alert alert-info'>
            <strong><i class="fas fa-lightbulb"></i> Interprétation :</strong> Plus le score est élevé, plus le risque de développer une maladie cardiaque est important. 
            Un score supérieur à 66 nécessite une attention médicale immédiate.
        </div>
        """, unsafe_allow_html=True)

# ========== BO3 : CLASSIFICATION MULTI-CLASSE ==========
elif page == "Classification Multi-classe":
    st.markdown("""
    <div class='dashboard-header'>
        <h1><i class="fas fa-layer-group"></i> Classification par Niveau de Risque</h1>
        <p>BO3 : Classification en 3 niveaux (Faible / Moyen / Élevé)</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='alert alert-info'>
        <strong><i class="fas fa-info-circle"></i> Objectif :</strong> Identifier les cinq facteurs les plus prédictifs et construire un modèle interprétable 
        capable de classer les patients selon leur niveau de risque (risque faible/moyen/élevé).
        <br><strong><i class="fas fa-tree"></i> Modèle :</strong> Random Forest Classifier
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    features = models['BO3']['features']
    top_features = models['BO3']['top_features']
    
    st.markdown(f"""
    <div class='content-card warning'>
        <h3 style='color: #ffae1f; margin-top: 0;'><i class="fas fa-search"></i> Top 5 Facteurs Prédictifs</h3>
        <div style='display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem;'>
            {' '.join([f"<span class='badge badge-warning'><i class='fas fa-star'></i> {i+1}. {feat}</span>" for i, feat in enumerate(top_features)])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='content-card primary'>
        <div class='card-header'>
            <h2 class='card-title'><i class="fas fa-edit"></i> Formulaire de Saisie</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=50, step=1, key="bo3_age")
        sysBP = st.number_input("Pression Systolique (mmHg)", min_value=80.0, max_value=250.0, 
                               value=120.0, step=0.1, key="bo3_sysBP")
        diaBP = st.number_input("Pression Diastolique (mmHg)", min_value=40.0, max_value=150.0, 
                               value=80.0, step=0.1, key="bo3_diaBP")
        totChol = st.number_input("Cholestérol Total (mg/dL)", min_value=100.0, max_value=400.0, 
                                 value=200.0, step=1.0, key="bo3_totChol")
    
    with col2:
        BMI = st.number_input("IMC (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, key="bo3_BMI")
        heartRate = st.number_input("Fréquence Cardiaque (bpm)", min_value=40, max_value=120, 
                                   value=75, step=1, key="bo3_heartRate")
        glucose = st.number_input("Glycémie (mg/dL)", min_value=50.0, max_value=300.0, 
                                 value=85.0, step=1.0, key="bo3_glucose")
        currentSmoker = st.selectbox("Fumeur Actuel", [0, 1], 
                                    format_func=lambda x: "Non" if x == 0 else "Oui", key="bo3_smoker")
    
    with col3:
        prevalentHyp = st.selectbox("Hypertension", [0, 1], 
                                   format_func=lambda x: "Non" if x == 0 else "Oui", key="bo3_hyp")
        diabetes = st.selectbox("Diabète", [0, 1], 
                               format_func=lambda x: "Non" if x == 0 else "Oui", key="bo3_diabetes")
    
    if st.button("Classer le Patient", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'age': [age], 'sysBP': [sysBP], 'diaBP': [diaBP], 'totChol': [totChol],
            'BMI': [BMI], 'heartRate': [heartRate], 'glucose': [glucose],
            'currentSmoker': [currentSmoker], 'prevalentHyp': [prevalentHyp], 'diabetes': [diabetes]
        })
        
        input_scaled = models['BO3']['scaler'].transform(input_data)
        prediction = models['BO3']['model'].predict(input_scaled)[0]
        probabilities = models['BO3']['model'].predict_proba(input_scaled)[0]
        
        risk_levels = {0: "FAIBLE", 1: "MOYEN", 2: "ÉLEVÉ"}
        risk_colors = {0: "#13deb9", 1: "#ffae1f", 2: "#fa896b"}
        risk_bgs = {0: "#15171a", 1: "#15171a", 2: "#15171a"}
        risk_icons = {0: "check-circle", 1: "exclamation-circle", 2: "exclamation-triangle"}
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            risk_label = risk_levels[prediction]
            st.markdown(f"""
            <div class='content-card' style='border-left: 4px solid {risk_colors[prediction]}; background: {risk_bgs[prediction]};'>
                <div style='text-align: center;'>
                    <div style='font-size: 3rem; color: {risk_colors[prediction]}; margin-bottom: 1rem;'><i class="fas fa-{risk_icons[prediction]}"></i></div>
                    <h3 style='color: {risk_colors[prediction]}; margin: 1rem 0;'>Niveau de Risque</h3>
                    <div style='font-size: 2.5rem; font-weight: 700; color: {risk_colors[prediction]};'>{risk_label}</div>
                    <div style='font-size: 1.5rem; font-weight: 600; color: {risk_colors[prediction]}; margin-top: 1rem;'>{probabilities[prediction]*100:.1f}% de confiance</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            fig = go.Figure(data=[
                go.Bar(
                    x=['Faible', 'Moyen', 'Élevé'],
                    y=[probabilities[0]*100, probabilities[1]*100, probabilities[2]*100],
                    marker=dict(color=['#13deb9', '#ffae1f', '#fa896b'], line=dict(color='white', width=2)),
                    text=[f'{probabilities[0]*100:.1f}%', f'{probabilities[1]*100:.1f}%', f'{probabilities[2]*100:.1f}%'],
                    textposition='auto',
                    textfont=dict(size=16, color='white', family='Arial Black')
                )
            ])
            fig.update_layout(
                title=dict(text="Probabilités par Niveau de Risque", font=dict(size=20, color='#2c3e50'), x=0.5),
                yaxis=dict(title="Probabilité (%)", range=[0, 100]),
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div class='content-card info'>
            <div class='card-header'>
                <h2 class='card-title'><i class="fas fa-chart-bar"></i> Importance des Facteurs (Top 5)</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': models['BO3']['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(5)
        
        fig2 = px.bar(
            feature_importance, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='Viridis',
            text='Importance'
        )
        fig2.update_traces(
            texttemplate='%{text:.3f}', textposition='outside',
            marker=dict(line=dict(color='white', width=2))
        )
        fig2.update_layout(
            title=dict(text="Top 5 Facteurs les Plus Prédictifs", font=dict(size=18, color='#2c3e50'), x=0.5),
            xaxis=dict(title="Importance"),
            yaxis=dict(title="Facteur"),
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

# ========== BO4 : CLUSTERING ==========
elif page == "Clustering":
    st.markdown("""
    <div class='dashboard-header'>
        <h1><i class="fas fa-project-diagram"></i> Groupes de Patients Similaires</h1>
        <p>BO4 : Clustering des patients en groupes similaires</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='alert alert-info'>
        <strong><i class="fas fa-info-circle"></i> Objectif :</strong> Former des clusters cohérents basés sur les données comportementales et cliniques 
        pour identifier des groupes de patients similaires.
        <br><strong><i class="fas fa-sitemap"></i> Modèle :</strong> K-Means Clustering (3 clusters)
    </div>
    """, unsafe_allow_html=True)
    
    models = load_models()
    features = models['BO4']['features']
    cluster_info = models['BO4']['cluster_info']
    
    st.markdown("""
    <div class='content-card danger'>
        <div class='card-header'>
            <h2 class='card-title'><i class="fas fa-edit"></i> Formulaire de Saisie</h2>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", min_value=18, max_value=100, value=50, step=1, key="bo4_age")
        sysBP = st.number_input("Pression Systolique (mmHg)", min_value=80.0, max_value=250.0, 
                               value=120.0, step=0.1, key="bo4_sysBP")
        totChol = st.number_input("Cholestérol Total (mg/dL)", min_value=100.0, max_value=400.0, 
                                 value=200.0, step=1.0, key="bo4_totChol")
    
    with col2:
        BMI = st.number_input("IMC (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, key="bo4_BMI")
        currentSmoker = st.selectbox("Fumeur Actuel", [0, 1], 
                                    format_func=lambda x: "Non" if x == 0 else "Oui", key="bo4_smoker")
        glucose = st.number_input("Glycémie (mg/dL)", min_value=50.0, max_value=300.0, 
                                 value=85.0, step=1.0, key="bo4_glucose")
    
    if st.button("Assigner au Cluster", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'age': [age], 'sysBP': [sysBP], 'totChol': [totChol],
            'BMI': [BMI], 'currentSmoker': [currentSmoker], 'glucose': [glucose]
        })
        
        input_scaled = models['BO4']['scaler'].transform(input_data)
        cluster = models['BO4']['model'].predict(input_scaled)[0]
        
        st.markdown("---")
        
        cluster_colors = ["#5d87ff", "#ffae1f", "#13deb9"]
        cluster_bgs = ["rgba(93, 135, 255, 0.1)", "rgba(255, 174, 31, 0.1)", "rgba(19, 222, 185, 0.1)"]
        
        st.markdown(f"""
        <div class='content-card' style='border-left: 4px solid {cluster_colors[cluster]}; background: #15171a;'>
            <div style='text-align: center;'>
                <div style='font-size: 3rem; color: {cluster_colors[cluster]}; margin-bottom: 1rem;'><i class="fas fa-users"></i></div>
                <h3 style='color: {cluster_colors[cluster]}; margin: 1rem 0;'>Le patient appartient au</h3>
                <div style='font-size: 3rem; font-weight: 700; color: {cluster_colors[cluster]};'>Cluster {cluster + 1}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <h3 style='color: #cfcfd0; margin-bottom: 1rem;'><i class="fas fa-info-circle"></i> Caractéristiques du Cluster {cluster + 1}</h3>
        """, unsafe_allow_html=True)
        cluster_characteristics = cluster_info.iloc[cluster]
        
        cols = st.columns(len(features))
        for idx, feature in enumerate(features):
            with cols[idx]:
                st.metric(feature, f"{cluster_characteristics[feature]:.1f}")
        
        st.markdown("""
        <div class='content-card success'>
            <div class='card-header'>
                <h2 class='card-title'><i class="fas fa-chart-line"></i> Comparaison des Clusters</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure()
        cluster_names = ['Cluster 1', 'Cluster 2', 'Cluster 3']
        colors = ['#5d87ff', '#ffae1f', '#13deb9']
        
        for i in range(len(cluster_info)):
            cluster_values = cluster_info.iloc[i]
            fig.add_trace(go.Scatter(
                x=features, y=cluster_values,
                mode='lines+markers',
                name=cluster_names[i],
                line=dict(width=4 if i == cluster else 2, color=colors[i]),
                marker=dict(size=12 if i == cluster else 8, color=colors[i])
            ))
        
        fig.update_layout(
            title=dict(text="Comparaison des Caractéristiques des Clusters", font=dict(size=20, color='#2c3e50'), x=0.5),
            xaxis=dict(title="Caractéristiques"),
            yaxis=dict(title="Valeur Moyenne"),
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        cluster_descriptions = {
            0: "**Cluster 1** : Patients avec profil de risque modéré",
            1: "**Cluster 2** : Patients avec profil de risque élevé",
            2: "**Cluster 3** : Patients avec profil de risque faible"
        }
        st.markdown(f"""
        <div class='alert alert-info'>
            <i class="fas fa-info-circle"></i> {cluster_descriptions[cluster]}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <h3 style='color: #cfcfd0; margin-bottom: 1rem;'><i class="fas fa-table"></i> Tableau Comparatif des Clusters</h3>
        """, unsafe_allow_html=True)
        display_cluster_info = cluster_info.copy()
        display_cluster_info.index = [f"Cluster {i+1}" for i in range(len(display_cluster_info))]
        st.dataframe(display_cluster_info.round(2), use_container_width=True)
