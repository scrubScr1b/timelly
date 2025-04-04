import streamlit as st
import pandas as pd

st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title("🧹 Data Cleaning Page")

# Load raw data
df = pd.read_excel("data/df_mz.xlsx")

st.subheader("📦 Raw Data Preview")
st.dataframe(df, use_container_width=True)

# Display data types
st.subheader("🧬 Data Types")
st.write(df.dtypes)

# Check for missing values
st.subheader("🚫 Missing Values")
st.write(df.isnull().sum())

# Convert 'Date' column to datetime if available
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    st.success("Kolom 'Date' berhasil dikonversi ke format datetime.")

# Add 'Year' and 'Month' columns if Date is valid
if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.strftime("%b")
    st.success("Kolom 'Year' dan 'Month' berhasil ditambahkan dari 'Date'.")

# Drop duplicates option
if st.checkbox("🧹 Hapus duplikat", value=True):
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    st.info(f"{before - after} duplikat dihapus.")

# Drop selected columns
st.subheader("🧯 Drop Kolom Tertentu")
cols_to_drop = st.multiselect("Pilih kolom yang ingin dihapus", df.columns)
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    st.warning(f"Kolom {cols_to_drop} telah dihapus.")

# Imputasi Nilai Kosong
st.subheader("🩹 Imputasi Nilai Kosong")
null_cols = df.columns[df.isnull().any()]
for col in null_cols:
    method = st.selectbox(f"Imputasi untuk '{col}'", ["Biarkan", "Mean", "Median", "Custom"], key=col)
    if method == "Mean":
        df[col].fillna(df[col].mean(), inplace=True)
    elif method == "Median":
        df[col].fillna(df[col].median(), inplace=True)
    elif method == "Custom":
        custom_value = st.number_input(f"Masukkan nilai untuk '{col}'", key=f"{col}_custom")
        df[col].fillna(custom_value, inplace=True)

# Filter interaktif berdasarkan Qty
if "Qty" in df.columns:
    st.subheader("🔎 Filter Data Interaktif")
    qty_filter = st.slider("Tampilkan data dengan Qty lebih dari:", min_value=0, max_value=int(df["Qty"].max()), value=0)
    df = df[df["Qty"] > qty_filter]

# Show cleaned data preview
st.subheader("✅ Cleaned Data Preview")
st.dataframe(df.head(20), use_container_width=True)

# Save cleaned data to CSV
if st.button("💾 Simpan Data Cleaning"):
    df.to_csv("data/cleaned_data.csv", index=False)
    st.success("Data berhasil disimpan ke 'cleaned_data.csv'")
