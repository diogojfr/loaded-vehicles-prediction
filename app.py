import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.api import ExponentialSmoothing
from datetime import timedelta

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Forecast: Total de Veículos Carregados", layout="centered")
st.title("Predição de Veículos Carregados")
# st.write("Upload o arquivo de dados.")

# -----------------------------
# File uploader
# -----------------------------
uploaded_file = st.file_uploader("Upload de arquivo CSV (`delivery_date`, `loaded_vehicles`)", type="csv")

if uploaded_file:
    # Load and clean data
    df = pd.read_csv(uploaded_file, parse_dates=["delivery_date"])
    df = df.sort_values("delivery_date")
    df = df.rename(columns={"delivery_date": "Data_Entrega", "loaded_vehicles": "Veiculos_Carregados"})

    st.subheader("Preview dos Dados")
    st.dataframe(df.head())

    # -----------------------------
    # Inputs
    # -----------------------------
    target = st.number_input("Defina a quantidade alvo de veículos carregados para a previsão", value=1_000_000, step=100_000)
    forecast_days = st.slider("Quantidade de dias para a predição", min_value=30, max_value=1095, value=365)

    # -----------------------------
    # Fit model and forecast
    # -----------------------------
    with st.spinner("Realizando a predição..."):
        model = ExponentialSmoothing(df['Veiculos_Carregados'], trend='add', seasonal=None)
        fit = model.fit()
        forecast = fit.forecast(forecast_days)

        last_date = df['Data_Entrega'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'Data_Entrega': forecast_dates, 'Veiculos_Carregados': forecast})

        full_df = pd.concat([df[['Data_Entrega', 'Veiculos_Carregados']], forecast_df], ignore_index=True)
        full_df['cumulative'] = full_df['Veiculos_Carregados'].cumsum()

    # -----------------------------
    # Target prediction
    # -----------------------------
    reached = full_df[full_df['cumulative'] >= target]
    if not reached.empty:
        reached_date = reached.iloc[0]['Data_Entrega'].date()
        st.success(f" O total de {target:,.0f} de veículos carregados será alcançado em **{reached_date}**.")
    else:
        st.warning("O valor alvo está fora do intervalo de predição.")

    # -----------------------------
    # Plot: Daily Loaded Vehicles
    # -----------------------------
    st.subheader("Quantidade Diária de Veículos Carregados")
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Scatter(x=full_df['Data_Entrega'], y=full_df['Veiculos_Carregados'],
                                   mode='lines', name='Carregados Diariamente',
                                   line=dict(color='blue')))
    fig_daily.update_layout(title='Veículos Carregados por Dia',
                            xaxis_title='Data',
                            yaxis_title='Veículos Carregados',
                            hovermode='x unified')
    st.plotly_chart(fig_daily, use_container_width=True)

    # -----------------------------
    # Plot: Cumulative Loaded Vehicles
    # -----------------------------
    st.subheader("Total Acumulado de Veículos Carregados")
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=full_df['Data_Entrega'], y=full_df['cumulative'],
                                 mode='lines', name='Acumulado',
                                 line=dict(color='green')))
    fig_cum.add_trace(go.Scatter(x=[full_df['Data_Entrega'].iloc[0], full_df['Data_Entrega'].iloc[-1]],
                                 y=[target, target],
                                 mode='lines', name=f'Alvo: {target:,.0f}',
                                 line=dict(color='red', dash='dash')))

    fig_cum.update_layout(title='Total Acumulado de Veículos Carregados no Tempo',
                          xaxis_title='Data',
                          yaxis_title='Acumulado de Veículos',
                          hovermode='x unified')
    st.plotly_chart(fig_cum, use_container_width=True)
else:
    st.info("Faça o upload do arquivo CSV. Colunas necessárias: `delivery_date`, `loaded_vehicles`.")
