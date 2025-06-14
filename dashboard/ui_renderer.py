import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import base64

class UIRenderer:
    def __init__(self):
        """
        Initialise le moteur de rendu d'interface Streamlit.
        """
        pass  # Aucun état à maintenir pour l'instant

    def render_sidebar(self, zones, horizons):
        """
        Affiche la barre latérale de configuration.

        :return: tuple (zone, horizon) sélectionnés
        """
        st.sidebar.title("Configuration")
        zone = st.sidebar.selectbox("Zone", zones)
        horizon = st.sidebar.selectbox("Horizon", horizons)
        return zone, horizon

    def render_date_selector(self, df):
        """
        Affiche les champs de sélection de plage de dates dans la sidebar.

        :param df: DataFrame dont l'index est de type DatetimeIndex
        :return: start_date, end_date sélectionnés
        """
        st.sidebar.subheader("Filtrage temporel")
        min_date = df.index.min()
        max_date = df.index.max()

        start_date = st.sidebar.date_input("Date de début", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.sidebar.date_input("Date de fin", value=max_date, min_value=start_date, max_value=max_date)

        return start_date, end_date

    def render_title(self, zone, horizon):
        """
        Affiche le titre principal de la page.
        """
        st.title(f"Prévisions de la demande — {zone.capitalize()} ({horizon})")

    def render_summary_statistics(self, df):
        """
        Affiche les statistiques descriptives de la demande réelle.
        """
        st.subheader("Statistiques de la demande réelle")
        st.write(df["demand"].describe())

    def render_model_info(self, model, name='Meilleur modèle'):
        """
        Affiche le type de modèle sélectionné.
        """
        st.subheader("Modèle sélectionné")
        st.write(f"Type de modèle : `{type(model).__name__}`, name: `{name}`")

    def render_predictions_chart(self, y_true, y_pred, index):
        """
        Affiche un graphique interactif comparant y_true et y_pred.
        """
        st.subheader("Comparaison Réel vs Prédiction")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_true, x=index, name="Réel", line=dict(color="royalblue")))
        fig.add_trace(go.Scatter(y=y_pred, x=index, name="Prédiction", line=dict(color="orange")))
        fig.update_layout(height=400, xaxis_title="Temps", yaxis_title="Demande")
        st.plotly_chart(fig, use_container_width=True)

    def render_performance_kpis(self, kpi_dict):
        """
        Affiche les indicateurs MAE, RMSE, MAPE.
        """
        st.subheader("Indicateurs de performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE",  f"{kpi_dict['MAE']:.2f}")
        col2.metric("RMSE", f"{kpi_dict['RMSE']:.2f}")
        col3.metric("MAPE", f"{kpi_dict['MAPE']:.2f}%")

    def render_download_button(self, y_true, y_pred, index):
        """
        Affiche un lien pour télécharger les prédictions au format CSV.
        """
        df_out = pd.DataFrame({
            "Date": index,
            "Réel": y_true,
            "Prédiction": y_pred
        })

        csv = df_out.to_csv(index=False).encode('utf-8')
        b64 = base64.b64encode(csv).decode()

        st.markdown(
            f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Télécharger les prédictions (CSV)</a>',
            unsafe_allow_html=True
        )

    def render_model_explanation(self, zone, horizon):
        """
        Affiche une explication textuelle simple du modèle.
        """
        st.subheader("Explication du modèle")
        st.info(f"""
        Ce modèle prédit la demande énergétique en Espagne, pour la zone **{zone}** avec une granularité **{horizon}**.
        Il a été sélectionné comme le meilleur modèle basé sur les métriques d'évaluation historiques.

        Des explications plus détaillées (SHAP, attention, importance des features) peuvent être ajoutées ici.
        Comme constaté, nos modèles sont mauvais pour les predictions journalières. Nous y travaillons
        pour rendre ces modèles performants.
        """)
