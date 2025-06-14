import ast
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("data"))
from tensorflow.keras.models import load_model
from src.models.dl_utils import scale_data
from src.models.evaluation import evaluate_regression

class ModelHandler:
    def __init__(self, model_dir="data/models/models"):
        self.model_dir = model_dir
        self.horizon = None
        self.model = None
        self.zone = None

    def load_model(self, zone: str, horizon: str):
        """
        Charge le meilleur modèle pour la zone et l’horizon donnés.

        :param zone: nom de la zone (ex: "madrid")
        :param horizon: granularité temporelle ("daily" ou "hourly")
        :return: modèle Keras chargé, ou None si introuvable
        """
        self.zone = zone
        self.horizon = horizon
        # Cherche tous les fichiers correspondant à la zone et horizon
        candidates = [f for f in os.listdir(self.model_dir)
                      if f.startswith(f"{zone}_{horizon}_") and f.endswith(".h5")]

        if not candidates:
            print(f"Aucun modèle trouvé pour {zone}_{horizon} dans {self.model_dir}")
            return None

        model_file = sorted(candidates)[0]
        model_path = os.path.join(self.model_dir, model_file)

        try:
            model = load_model(model_path, compile= False)
            self.model = model
            print(f"Modèle chargé : {model_file}")
            return model
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_file} : {e}")
            return None

    def get_features_for_file(self, TARGET_COL='demand'):
        df1 = pd.read_csv(f'data/submission/features_selected_{self.horizon}.csv')
        
        df1['features'] = df1['features'].apply(ast.literal_eval)
        
        features = df1.loc[df1['zone'] == self.zone, 'features'].values[0]
        return [col for col in features if col != TARGET_COL]
    def create_sequences(self, df, target_col, lookback, features):
        data = df[features]
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data.iloc[i - lookback:i].values)
            y.append(df.iloc[i][target_col])
        return np.array(X), np.array(y)
    def predict(self, df: pd.DataFrame):
        """
        Prédit la demande sur la portion test de la série temporelle.
        
        :param df: DataFrame complet (avec colonne "demand")
        :return: (y_true, y_pred, index_test)
        """

        features = self.get_features_for_file()
        split = int(len(df)*0.8)
        df_train, df_test = df.iloc[:split], df.iloc[split:]
        LOOKBACK = 24 if self.horizon=="hourly" else 7
        X_seq, y_seq = self.create_sequences(df, 'demand', LOOKBACK, features)
        n_test = len(df_test)
        X_test_seq = X_seq[-n_test:]
        _, X_test_seq_scaled, _ = scale_data(X_seq[:-n_test], X_test_seq)
        y_true = y_seq[-n_test:].ravel()
        y_pred = self.model.predict(X_test_seq_scaled).ravel()
        idx = df_test.index

        return y_true, y_pred, idx


    def evaluate(self, y_true, y_pred):
        """
        Calcule les KPI de régression (MAE, RMSE, MAPE).
        
        :param y_true: valeurs réelles
        :param y_pred: valeurs prédites
        :return: dictionnaire {MAE, RMSE, MAPE}
        """
        return evaluate_regression(None, None, pd.Series(y_true), pd.Series(y_pred))
