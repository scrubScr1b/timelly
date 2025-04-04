import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Forecasting", layout="wide")
st.title("📈 Sales Forecasting (Regresi Linear Berganda)")

# Load data
df = pd.read_excel("data/df_mz.xlsx")

# Pastikan kolom Date dalam format datetime
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%B")

# Sidebar filter
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df["Year"].unique(), reverse=True))
selected_month = st.sidebar.selectbox("Pilih Bulan", sorted(df["Month"].unique()))

# Filter data untuk bulan yang dipilih
df_filtered = df[(df["Year"] == selected_year) & (df["Month"] == selected_month)]

if df_filtered.empty:
    st.warning("Data tidak ditemukan untuk bulan dan tahun yang dipilih.")
else:
    # Agregasi per bulan
    monthly_data = df_filtered.groupby(["Year", "Month"]).agg(
        Total_Transactions=("No_Invoice", "nunique"),  # x1
        Total_Qty=("Qty", "sum"),                     # x2
        Total_Sales=("Total_Sales", "sum")            # y
    ).reset_index()

    # Model regresi
    X = monthly_data[["Total_Transactions", "Total_Qty"]]
    y = monthly_data["Total_Sales"]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    # Evaluasi model
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))  # Tanpa squared=False
    mae = mean_absolute_error(y, y_pred)

    st.subheader("📊 Model Performance Metrics")
    metrics = {
        "R² Score": r2,
        "RMSE": rmse,
        "MAE": mae
    }
    st.dataframe(pd.DataFrame(metrics, index=["Metrics"]).T)

    # Prediksi bulan berikutnya
    next_month = selected_month + 1 if selected_month < 12 else 1
    next_year = selected_year if selected_month < 12 else selected_year + 1
    st.markdown(f"### Prediksi Omset untuk Bulan Berikutnya: **{next_month:02}/{next_year}**")

    df_next = df[(df["Year"] == next_year) & (df["Month"] == next_month)]
    if df_next.empty:
        st.warning("Tidak ada data historis untuk bulan berikutnya.")
    else:
        forecast_monthly = df_next.groupby(["Year", "Month"]).agg(
            Total_Transactions=("No_Invoice", "nunique"),
            Total_Qty=("Qty", "sum")
        ).reset_index()

        X_pred = sm.add_constant(forecast_monthly[["Total_Transactions", "Total_Qty"]])
        forecast_monthly["Predicted_Sales"] = model.predict(X_pred)

        total_forecast = forecast_monthly["Predicted_Sales"].sum()
        st.success(f"🧮 Total prediksi omset bulan {next_month:02}/{next_year}: **Rp {total_forecast:,.0f}**")

        st.subheader("Detail Prediksi Bulanan")
        st.dataframe(forecast_monthly)

        # Visualisasi prediksi
        fig = px.bar(
            forecast_monthly,
            x="Month",
            y="Predicted_Sales",
            labels={"Predicted_Sales": "Prediksi Omset"},
            title="📉 Prediksi Omset Bulan Berikutnya",
            text_auto=".2s"
        )
        st.plotly_chart(fig, use_container_width=True)
