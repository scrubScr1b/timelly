import streamlit as st
import pandas as pd
from utils import load_saved_dataset 

st.title("Data Preview")

# Cek apakah data tersedia, jika belum coba load dari file
if "data" not in st.session_state:
    df, source = load_saved_dataset()
    if df is not None:
        st.session_state["data"] = df
        st.session_state["source"] = source
    else:
        if st.session_state.get("role") == "admin":
            st.warning("Silakan upload file terlebih dahulu di halaman admin")
        else:
            st.warning("Admin Belum Mengupload File Dataset!")
        st.stop()

# Ambil data dari session state
df = st.session_state["data"]

# Tampilkan data langsung
st.dataframe(df)

# ====================
# 🔍 Statistik Deskriptif Dataset
# ====================
st.markdown("---")
st.subheader("Statistik Deskriptif Dataset")

st.write(f"Jumlah baris: **{df.shape[0]:,}**")
st.write(f"Jumlah kolom: **{df.shape[1]:,}**")
st.write("**Tipe Data per Kolom:**")
st.dataframe(pd.DataFrame(df.dtypes, columns=["Tipe Data"]))

st.write("**Deskripsi Statistik Numerik**")
numeric_cols = df.select_dtypes(include=["int", "float"]).columns
if not numeric_cols.empty:
    st.dataframe(df[numeric_cols].describe(
    ).transpose().style.format("{:,.2f}"))
else:
     st.info("Tidak ada kolom numerik dalam dataset.")

st.write("**Nilai Unik dan Kosong per Kolom**")
col_summary = pd.DataFrame({
        "Tipe Data": df.dtypes,
        "Jumlah Unik": df.nunique(),
        "Jumlah Kosong": df.isna().sum(),
        "% Kosong": (df.isna().mean() * 100).round(2)
    })
st.dataframe(col_summary)
