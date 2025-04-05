import streamlit as st
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Forecasting", layout="wide")
st.title("📈 Sales Forecasting per Customer (Linear Regression, Random Forest)")

st.markdown("""
**Variabel yang Digunakan dalam Model:**
- **X1 (Total Transactions):** Jumlah invoice unik (transaksi) per bulan untuk setiap customer.
- **X2 (Total Qty):** Total kuantitas produk yang terjual per bulan untuk setiap customer.
- **Y (Total Sales):** Total nilai penjualan (omset) per bulan untuk setiap customer.

Model ini bertujuan untuk memprediksi *Y (Total Sales)* berdasarkan variabel input *X1* dan *X2*, secara terpisah untuk setiap customer.
""")

# Load data
df = pd.read_excel("data/df_mz.xlsx")

# Format kolom tanggal
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%B")

# Sidebar filter
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df["Year"].unique(), reverse=True))
selected_month = st.sidebar.selectbox("Pilih Bulan", sorted(df["Month"].unique()))
selected_customer = st.sidebar.selectbox("Pilih Customer", sorted(df["Customers"].unique()))

# Filter data
df_filtered = df[(df["Year"] == selected_year) & (df["Month"] == selected_month) & (df["Customers"] == selected_customer)]

if df_filtered.empty:
    st.warning("Data tidak ditemukan untuk customer, bulan, dan tahun yang dipilih.")
else:
    # Agregasi bulanan untuk model per customer
    train = df[df["Customers"] == selected_customer].groupby(["Year", "Month"]).agg(
        Total_Transactions=("No_Invoice", "nunique"),
        Total_Qty=("Qty", "sum"),
        Total_Sales=("Total_Sales", "sum")
    ).reset_index()

    # Prediksi bulan berikutnya
    next_month = selected_month + 1 if selected_month < 12 else 1
    next_year = selected_year if selected_month < 12 else selected_year + 1

    test = df[(df["Year"] == next_year) & (df["Month"] == next_month) & (df["Customers"] == selected_customer)]
    if test.empty:
        st.warning("Tidak ada data historis untuk bulan berikutnya untuk customer ini.")
    else:
        test_agg = test.groupby(["Year", "Month"]).agg(
            Total_Transactions=("No_Invoice", "nunique"),
            Total_Qty=("Qty", "sum"),
            Total_Sales=("Total_Sales", "sum")
        ).reset_index()

        # Model Regresi Linear
        train_filtered = train[(train["Year"] < next_year) | ((train["Year"] == next_year) & (train["Month"] < next_month))]

        X_train = train_filtered[["Total_Transactions", "Total_Qty"]]
        y_train = train_filtered["Total_Sales"]
        X_train_sm = sm.add_constant(X_train)
        model_ols = sm.OLS(y_train, X_train_sm).fit()

        # Prediksi
        X_test_sm = sm.add_constant(test_agg[["Total_Transactions", "Total_Qty"]])
        test_agg["Predicted_Linear"] = model_ols.predict(X_test_sm)

        # Evaluasi
        y_pred_linear = model_ols.predict(X_train_sm)
        linear_metrics = {
            "R²": r2_score(y_train, y_pred_linear),
            "MAE": mean_absolute_error(y_train, y_pred_linear),
            "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_linear))
        }

        # Random Forest
        rf_model = RandomForestRegressor(random_state=42)
        rf_model.fit(X_train, y_train)
        test_agg["Predicted_RF"] = rf_model.predict(test_agg[["Total_Transactions", "Total_Qty"]])

        rf_pred_train = rf_model.predict(X_train)
        rf_metrics = {
            "R²": r2_score(y_train, rf_pred_train),
            "MAE": mean_absolute_error(y_train, rf_pred_train),
            "RMSE": np.sqrt(mean_squared_error(y_train, rf_pred_train))
        }

        # Output
        st.markdown(f"### Prediksi Omset Bulan Berikutnya untuk {selected_customer}: {next_month:02}/{next_year}")
        st.dataframe(test_agg[["Year", "Month", "Total_Transactions", "Total_Qty", "Total_Sales", "Predicted_Linear", "Predicted_RF"]])

        st.subheader("🔍 Model Evaluation Metrics")
        st.markdown("**Regresi Linear**")
        st.write(pd.DataFrame(linear_metrics, index=["Regresi Linear"]).T)

        st.markdown("**Random Forest**")
        st.write(pd.DataFrame(rf_metrics, index=["Random Forest"]).T)

        # Prediksi historis
        st.subheader("📅 Performa Prediksi Regresi Linear Tiap Bulan")
        historical_preds = train_filtered.copy()
        historical_preds["Predicted_Linear"] = model_ols.predict(sm.add_constant(historical_preds[["Total_Transactions", "Total_Qty"]]))
        historical_preds["Selisih"] = historical_preds["Total_Sales"] - historical_preds["Predicted_Linear"]
        st.dataframe(historical_preds[["Year", "Month", "Total_Sales", "Predicted_Linear", "Selisih"]])

        # Visualisasi Bar Chart hanya untuk bulan yang dipilih
        st.subheader("📊 Visualisasi Perbandingan Prediksi vs Aktual (Bar Chart)")
        current_data = historical_preds[historical_preds["Month"] == selected_month]
        current_data["Label"] = current_data["Year"].astype(str) + "-" + current_data["Month"].astype(str).str.zfill(2)

        current_data["Predicted_RF"] = rf_model.predict(current_data[["Total_Transactions", "Total_Qty"]])
        melted_df = current_data.melt(id_vars=["Label"], 
                               value_vars=["Total_Sales", "Predicted_Linear", "Predicted_RF"], 
                               var_name="Sumber", value_name="Sales")
        

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=melted_df, x="Label", y="Sales", hue="Sumber", ax=ax)
        ax.set_ylabel("Omset Penjualan")
        ax.set_xlabel("Bulan")
        ax.set_title(f"Perbandingan Omset Aktual vs Prediksi - {selected_customer} ({selected_month:02}/{selected_year})")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.0f", label_type="edge", fontsize=8)
        st.pyplot(fig)
