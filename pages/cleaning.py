# import streamlit as st
# import pandas as pd

# st.title("Data Cleaning Page")

# # Cek apakah data tersedia
# if "data" not in st.session_state:
#     st.warning("Silakan upload file terlebih dahulu di halaman utama.")
#     st.stop()

# df = st.session_state["data"]

# # Lanjutkan analisis seperti biasa
# st.write("Data tersedia:", df.shape)

# st.subheader("Raw Data Preview")
# st.dataframe(df, use_container_width=True)

# # Display data types
# st.subheader("Data Types")
# st.write(df.dtypes)

# # Check for missing values
# st.subheader("Missing Values")
# st.write(df.isnull().sum())

# # Convert 'Date' column to datetime if available
# if "Date" in df.columns:
#     df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
#     st.success("Kolom 'Date' berhasil dikonversi ke format datetime.")

# # Add 'Year' and 'Month' columns if Date is valid
# if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
#     df["Year"] = df["Date"].dt.year
#     df["Month"] = df["Date"].dt.strftime("%b")
#     st.success("Kolom 'Year' dan 'Month' berhasil ditambahkan dari 'Date'.")

# # Drop duplicates option
# if st.checkbox("Hapus duplikat", value=True):
#     before = len(df)
#     df = df.drop_duplicates()
#     after = len(df)
#     st.info(f"{before - after} duplikat dihapus.")

# # Drop selected columns
# st.subheader("Drop Kolom Tertentu")
# cols_to_drop = st.multiselect("Pilih kolom yang ingin dihapus", df.columns)
# if cols_to_drop:
#     df.drop(columns=cols_to_drop, inplace=True)
#     st.warning(f"Kolom {cols_to_drop} telah dihapus.")


# # Ubah Tipe Data Kolom
# st.subheader("Ubah Tipe Data Kolom")
# col_to_convert = st.selectbox("Pilih kolom yang ingin diubah tipenya", df.columns)

# type_options = ["int", "float", "str", "datetime"]
# selected_type = st.selectbox("Pilih tipe data baru", type_options)

# if st.button("Ubah Tipe Data"):
#     try:
#         if selected_type == "int":
#             df[col_to_convert] = df[col_to_convert].astype(int)
#         elif selected_type == "float":
#             df[col_to_convert] = df[col_to_convert].astype(float)
#         elif selected_type == "str":
#             df[col_to_convert] = df[col_to_convert].astype(str)
#         elif selected_type == "datetime":
#             df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors="coerce")
#         st.success(f"Tipe data kolom '{col_to_convert}' berhasil diubah menjadi {selected_type}.")
#     except Exception as e:
#         st.error(f"Gagal mengubah tipe data: {e}")
        
# # Show cleaned data preview
# st.subheader("Cleaned Data Preview")
# st.dataframe(df.head(20), use_container_width=True)

# # Save cleaned data to CSV
# if st.button("Simpan Data Cleaning"):
#     df.to_csv("data/cleaned_data.csv", index=False)
#     st.success("Data berhasil disimpan ke 'cleaned_data.csv'")

import pandas as pd
import streamlit as st

# Cek apakah ada dataset di session state
if "data" not in st.session_state:
    st.error("Dataset belum di-upload! Harap upload dataset terlebih dahulu.")
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

