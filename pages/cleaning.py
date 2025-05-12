# import pandas as pd
# import streamlit as st

# st.title("Data Cleaning")

# # Cek apakah data tersedia
# if "data" not in st.session_state:
#     if st.session_state.get("role") == "admin":
#         st.warning("Silakan upload file terlebih dahulu di halaman admin")
#     else:
#         st.warning("Admin belum upload file.")
#     st.stop()

# # Ambil dataset dari session state
# df = st.session_state["data"]

# # Tampilkan dataset awal
# st.subheader("Dataset Sebelum Cleaning")
# st.dataframe(df)

# # Fungsi untuk mengubah tipe data kolom
# def convert_column_type(col_name, new_type):
#     try:
#         if new_type == "int":
#             # Bersihkan data sebelum ubah ke integer (hapus non-numerik)
#             df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
#             df[col_name] = df[col_name].fillna(0).astype(int)
#         elif new_type == "float":
#             # Bersihkan data sebelum ubah ke float
#             df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
#             df[col_name] = df[col_name].fillna(0).astype(float)
#         elif new_type == "string":
#             # Ubah ke string
#             df[col_name] = df[col_name].astype(str)
#         st.success(f"Tipe data kolom '{col_name}' berhasil diubah menjadi {new_type}.")
#     except Exception as e:
#         st.error(f"Gagal mengubah tipe data kolom '{col_name}': {str(e)}")

# # Fungsi untuk menghapus kolom
# def remove_column(col_name):
#     if col_name in df.columns:
#         df.drop(columns=[col_name], inplace=True)
#         st.success(f"Kolom '{col_name}' berhasil dihapus.")
#     else:
#         st.error(f"Kolom '{col_name}' tidak ditemukan!")

# # Tampilkan kolom yang ada
# st.subheader("Kolom yang Tersedia di Dataset")
# st.write(df.columns.tolist())

# # Pilih kolom dan tipe data baru
# st.subheader("Pilih Kolom untuk Dikonversi")
# col_to_convert = st.selectbox("Pilih kolom", df.columns.tolist())
# new_type = st.selectbox("Pilih tipe data baru", ["int", "float", "string"])

# # Tombol untuk mengubah tipe data
# if st.button("Ubah Tipe Data"):
#     convert_column_type(col_to_convert, new_type)

# # Pilih kolom untuk dihapus
# st.subheader("Hapus Kolom")
# col_to_remove = st.selectbox("Pilih kolom untuk dihapus", df.columns.tolist())

# # Tombol untuk menghapus kolom
# if st.button("Hapus Kolom"):
#     remove_column(col_to_remove)

# # Tampilkan dataset setelah perubahan
# st.subheader("Dataset Setelah Cleaning")
# st.dataframe(df)

# # Simpan dataset yang telah dibersihkan
# if st.button("Simpan Dataset"):
#     # Simpan dataset ke session state
#     st.session_state["data"] = df
#     st.success("Dataset berhasil disimpan!")

# V2
import pandas as pd
import streamlit as st
import time

st.title("Data Cleaning")

# Cek apakah data tersedia
if "data" not in st.session_state:
    if st.session_state.get("role") == "admin":
        st.warning("Silakan upload file terlebih dahulu di halaman admin")
    else:
        st.warning("Admin belum upload file.")
    st.stop()

# Inisialisasi list kolom yang dihapus jika belum ada
if "deleted_columns" not in st.session_state:
    st.session_state["deleted_columns"] = {}

# Ambil dataset dari session state
df = st.session_state["data"]

# Tampilkan dataset awal
st.subheader("Preview isi Dataset")
st.dataframe(df)

# ===============================
# TABEL TIPE DATA KOLUMN
# ===============================
st.subheader("Informasi Tipe Data Kolom")
dtype_df = pd.DataFrame({
    "Kolom": df.columns,
    "Tipe Data": df.dtypes.astype(str).values
})
st.dataframe(dtype_df, use_container_width=True)

# ===============================
# KONVERSI TIPE DATA
# ===============================
st.subheader("Pilih Kolom untuk Dikonversi")
col_to_convert = st.selectbox("Pilih kolom", df.columns.tolist(), key="convert_col")
new_type = st.selectbox("Pilih tipe data baru", ["int", "float", "string"])
confirm_convert = st.checkbox("Saya yakin ingin mengubah tipe data kolom ini")
convert_submit = st.button("Ubah Tipe Data")

if convert_submit:
    if confirm_convert:
        with st.spinner(f"Mengubah tipe data kolom '{col_to_convert}' ke {new_type}..."):
            time.sleep(1.5)

            try:
                if new_type == "int":
                    df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors="coerce").fillna(0).astype(int)
                elif new_type == "float":
                    df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors="coerce").fillna(0).astype(float)
                elif new_type == "string":
                    df[col_to_convert] = df[col_to_convert].astype(str)

                # Simpan perubahan
                st.session_state["data"] = df
                st.success(f"Tipe data kolom '{col_to_convert}' berhasil diubah menjadi {new_type}.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Gagal mengubah tipe data kolom '{col_to_convert}': {str(e)}")
    else:
        st.warning("Silakan centang konfirmasi sebelum mengubah tipe data.")

# ===============================
# HAPUS KOLOM
# ===============================
st.subheader("Hapus Kolom")
col_to_remove = st.selectbox("Pilih kolom untuk dihapus", df.columns.tolist(), key="remove_col")
confirm_delete = st.checkbox("Saya yakin ingin menghapus kolom ini")
delete_submit = st.button("Hapus Kolom")

if delete_submit:
    if confirm_delete:
        if col_to_remove in df.columns:
            with st.spinner(f"Menghapus kolom '{col_to_remove}'..."):
                time.sleep(1.5)
                # Simpan isi kolom sebelum dihapus
                st.session_state["deleted_columns"][col_to_remove] = df[col_to_remove].copy()
                df.drop(columns=[col_to_remove], inplace=True)

                # Simpan perubahan
                st.session_state["data"] = df
                st.success(f"Kolom '{col_to_remove}' berhasil dihapus!")
                time.sleep(1)
                st.rerun()
        else:
            st.error(f"Kolom '{col_to_remove}' tidak ditemukan!")
    else:
        st.warning("Silakan centang konfirmasi sebelum menghapus kolom.")

# ===============================
# Kembalikan Kolom yang Dihapus
# ===============================
st.subheader("Kembalikan Kolom yang Dihapus")
if st.session_state["deleted_columns"]:
    col_to_restore = st.selectbox("Pilih kolom yang ingin dikembalikan", list(st.session_state["deleted_columns"].keys()))
    if st.button("Kembalikan Kolom"):
        df[col_to_restore] = st.session_state["deleted_columns"][col_to_restore]
        df = df[df.columns.sort_values()]  # Optional: sort columns
        del st.session_state["deleted_columns"][col_to_restore]
        st.session_state["data"] = df
        st.success(f"Kolom '{col_to_restore}' berhasil dikembalikan!")
        st.rerun()
else:
    st.info("Tidak ada kolom yang bisa dikembalikan.")
