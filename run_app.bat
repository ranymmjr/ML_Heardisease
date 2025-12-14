@echo off
echo ========================================
echo  Systeme de Prediction du Risque Cardiaque
echo ========================================
echo.
echo Verification des modeles...
if not exist "model_BO1.pkl" (
    echo Les modeles n'existent pas. Entrainement en cours...
    python train_models.py
    echo.
)
echo.
echo Lancement de l'application Streamlit...
streamlit run app.py
pause

