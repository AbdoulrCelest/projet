import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model as load_keras

BEST_CSV      = Path("outputs/reports/all_zones_dl_metrics.csv")
MODELS_DIR    = Path("models")
DL_MODELS_DIR = Path("models/dl")

def load_best_table() -> pd.DataFrame:
    """
    Charge le tableau CSV contenant les performances
    des modèles pour chaque zone et horizon.
    """
    df = pd.read_csv(BEST_CSV)
    # Remplacer les chaînes 'inf' ou 'inf.' par np.inf
    df = df.replace({'inf': float('inf'), 'inf.': float('inf')})
    return df

def get_model_path(zone: str, horizon: str, model_name: str) -> Path:
    """
    Construit le chemin vers le fichier du modèle sauvegardé.
    """
    model_name_clean = model_name.lower().replace("-", "")
    if model_name in ("ElasticNet", "RandomForest", "LightGBM", "Ridge", "ARIMA"):
        if model_name == "RandomForest":
            model_name_clean = "rf"
        filename = f"{zone}_{horizon}_{model_name_clean}.pkl"
        return MODELS_DIR / filename
    else:
        filename = f"{zone}_{horizon}_{model_name}.h5"
        return DL_MODELS_DIR / filename

def load_best_model(zone: str, horizon: str):
    """
    Charge le modèle pour une zone et un horizon
    """
    
