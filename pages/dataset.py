import os
import pandas as pd
import streamlit as st

UPLOAD_DIR = "uploaded_data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

CSV_PATH = os.path.join(UPLOAD_DIR, "dataset.csv")
EXCEL_PATH = os.path.join(UPLOAD_DIR, "dataset.xlsx")

st.title("Upload Dataset")


def load_saved_dataset():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        source = "CSV"
    elif os.path.exists(EXCEL_PATH):
        df = pd.read_excel(EXCEL_PATH)
        source = "Excel"
    else:
        return False

    df.columns = df.columns.str.strip().str.lower()
    st.session_state["data"] = df
    st.session_state["source"] = source
    st.success(f"Dataset berhasil dimuat dari file {source} yang tersimpan!")
    return True


def upload_dataset():
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file:
        filename = uploaded_file.name
        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            df.to_csv(CSV_PATH, index=False)
            st.session_state["source"] = "CSV"
        else:
            df = pd.read_excel(uploaded_file)
            df.to_excel(EXCEL_PATH, index=False)
            st.session_state["source"] = "Excel"

        df.columns = df.columns.str.strip().str.lower()
        st.session_state["data"] = df
        st.success("Dataset berhasil diupload dan disimpan!")
        st.write("Kolom yang terbaca:", df.columns.tolist())
        st.rerun()


def delete_dataset():
    deleted = False
    for path in [CSV_PATH, EXCEL_PATH]:
        if os.path.exists(path):
            os.remove(path)
            deleted = True
    if deleted:
        st.session_state.pop("data", None)
        st.session_state.pop("source", None)
        st.success("Dataset berhasil dihapus!")
        st.rerun()
    else:
        st.warning("Tidak ada dataset yang bisa dihapus.")


# Load data dari file jika belum ada di session_state
if "data" not in st.session_state:
    load_saved_dataset()

# Jika data sudah tersedia
if "data" in st.session_state:
    df = st.session_state["data"]
    st.write(f"Dataset dari sumber: {st.session_state.get('source', '-')}")
    st.write(f"Jumlah baris: {df.shape[0]}, Jumlah kolom: {df.shape[1]}")
    st.write("Kolom yang terbaca:", df.columns.tolist())

    with st.expander("Preview 5 baris teratas"):
        st.dataframe(df.head())

    if st.button("Hapus Dataset"):
        delete_dataset()

else:
    upload_dataset()
