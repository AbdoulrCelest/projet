import yaml
import streamlit as st
import pandas as pd
import sys
import os
from dashboard.data_manager import DataManager
from dashboard.model_handler import ModelHandler
from dashboard.ui_renderer import UIRenderer

sys.path.append(os.path.abspath("dashboard"))
sys.path.append(os.path.abspath("src"))
sys.path.append(os.path.abspath("outputs"))

class DashboardApp:
    def __init__(self):
        """Initialise les composants principaux de l'application."""
        self.config = self.load_config()
        self.feature_cols = []
        self.data_manager = DataManager(self.config)
        self.model_handler = ModelHandler()
        self.ui_renderer = UIRenderer()

    def load_config(self):
        with open("config.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def run(self):
        """Point d’entrée principal pour afficher l'application Streamlit."""
        st.set_page_config(
            page_title="Prévision Énergie 🇪🇸",
            page_icon="⚡", layout="wide")

        best_df = pd.read_csv("outputs/reports/best_models.csv")
        zones = best_df['zone'].tolist()
        horizons = ["daily", "hourly"]

        zone, horizon = self.ui_renderer.render_sidebar(zones, horizons)

        df = self.data_manager.load_data(zone, horizon)

        start_date, end_date = self.ui_renderer.render_date_selector(df)
        df = self.data_manager.filter_by_date_range(start_date, end_date)

        self.ui_renderer.render_title(zone, horizon)
        self.ui_renderer.render_summary_statistics(df)

        model = self.model_handler.load_model(zone, horizon)

        if model is not None:
            self.ui_renderer.render_model_info(model)

            y_true, y_pred, index = self.model_handler.predict(df)
            self.ui_renderer.render_predictions_chart(y_true, y_pred, index)
            self.ui_renderer.render_download_button(y_true, y_pred, index)

            kpi_dict = self.model_handler.evaluate(y_true, y_pred)
            self.ui_renderer.render_performance_kpis(kpi_dict)

            self.ui_renderer.render_model_explanation(zone, horizon)
