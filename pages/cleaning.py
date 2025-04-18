import pandas as pd
import streamlit as st

st.title("Data Cleaning")

# Cek apakah data tersedia
if "data" not in st.session_state:
    if st.session_state.get("role") == "admin":
        st.warning("Silakan upload file terlebih dahulu di halaman admin")
    else:
        st.warning("Admin belum upload file.")
    st.stop()

# Ambil dataset dari session state
df = st.session_state["data"]

# Tampilkan dataset awal
st.subheader("Dataset Sebelum Cleaning")
st.dataframe(df)

# Fungsi untuk mengubah tipe data kolom
def convert_column_type(col_name, new_type):
    try:
        if new_type == "int":
            # Bersihkan data sebelum ubah ke integer (hapus non-numerik)
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            df[col_name] = df[col_name].fillna(0).astype(int)
        elif new_type == "float":
            # Bersihkan data sebelum ubah ke float
            df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
            df[col_name] = df[col_name].fillna(0).astype(float)
        elif new_type == "string":
            # Ubah ke string
            df[col_name] = df[col_name].astype(str)
        st.success(f"Tipe data kolom '{col_name}' berhasil diubah menjadi {new_type}.")
    except Exception as e:
        st.error(f"Gagal mengubah tipe data kolom '{col_name}': {str(e)}")

# Fungsi untuk menghapus kolom
def remove_column(col_name):
    if col_name in df.columns:
        df.drop(columns=[col_name], inplace=True)
        st.success(f"Kolom '{col_name}' berhasil dihapus.")
    else:
        st.error(f"Kolom '{col_name}' tidak ditemukan!")

# Tampilkan kolom yang ada
st.subheader("Kolom yang Tersedia di Dataset")
st.write(df.columns.tolist())

# Pilih kolom dan tipe data baru
st.subheader("Pilih Kolom untuk Dikonversi")
col_to_convert = st.selectbox("Pilih kolom", df.columns.tolist())
new_type = st.selectbox("Pilih tipe data baru", ["int", "float", "string"])

# Tombol untuk mengubah tipe data
if st.button("Ubah Tipe Data"):
    convert_column_type(col_to_convert, new_type)

# Pilih kolom untuk dihapus
st.subheader("Hapus Kolom")
col_to_remove = st.selectbox("Pilih kolom untuk dihapus", df.columns.tolist())

# Tombol untuk menghapus kolom
if st.button("Hapus Kolom"):
    remove_column(col_to_remove)

# Tampilkan dataset setelah perubahan
st.subheader("Dataset Setelah Cleaning")
st.dataframe(df)

# Simpan dataset yang telah dibersihkan
if st.button("Simpan Dataset"):
    # Simpan dataset ke session state
    st.session_state["data"] = df
    st.success("Dataset berhasil disimpan!")

