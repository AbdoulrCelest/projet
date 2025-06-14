# dashboard/components.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import base64
import datetime


def display_title(zone: str, horizon: str):
    st.title(f"Prévisions de la demande — {zone.capitalize()} ({horizon})")


def display_summary_statistics(df: pd.DataFrame):
    st.subheader("Statistiques de la demande réelle")
    st.write(df["demand"].describe())


def display_model_type(model):
    st.subheader("Modèle sélectionné")
    st.write(f"Type de modèle : `{type(model).__name__}`")


def display_predictions_chart(y_true, y_pred, index):
    st.subheader("Comparaison Réel vs Prédiction")

    df_chart = pd.DataFrame({
        "Réel": y_true,
        "Prédiction": y_pred
    }, index=index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, x=index, name="Réel", line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(y=y_pred, x=index, name="Prédiction", line=dict(color="orange")))
    fig.update_layout(height=400, xaxis_title="Temps", yaxis_title="Demande")

    st.plotly_chart(fig, use_container_width=True)

def display_kpi(kpi_dict):
    st.subheader("Indicateurs de performance")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE",  f"{kpi_dict['MAE']:.2f}")
    col2.metric("RMSE", f"{kpi_dict['RMSE']:.2f}")
    col3.metric("MAPE", f"{kpi_dict['MAPE']:.2f}%")

def date_range_selector(df):
    st.sidebar.subheader("Filtrage temporel")

    min_date = df.index.min()
    max_date = df.index.max()

    start_date = st.sidebar.date_input("Date de début", value=min_date, min_value=min_date, max_value=max_date)
    end_date   = st.sidebar.date_input("Date de fin", value=max_date, min_value=start_date, max_value=max_date)

    if isinstance(df.index, pd.DatetimeIndex):
        mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
        return df.loc[mask]
    else:
        return df


def download_predictions_button(y_true, y_pred, index):
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


def display_model_explanation(zone, horizon):
    st.subheader("🔍 Explication du modèle")
    st.info(f"""
    Ce modèle prédit la demande énergétique pour la zone **{zone}** avec une granularité **{horizon}**.
    Il a été sélectionné comme le meilleur modèle basé sur les métriques d'évaluation historiques.
    
    Des explications plus détaillées (SHAP, attention, importance des features) peuvent être ajoutées ici.
    """)
