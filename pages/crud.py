import streamlit as st
import pandas as pd
import os

# Fungsi untuk load data pengguna
def load_users():
    if not os.path.exists("data/users.csv"):
        pd.DataFrame(columns=["username", "password", "role"]).to_csv("data/users.csv", index=False)
    return pd.read_csv("data/users.csv")

# Fungsi untuk menyimpan pengguna baru
def save_user(username, password, role="user"):
    users = load_users()
    new_user = pd.DataFrame([[username, password, role]], columns=["username", "password", "role"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("data/users.csv", index=False)

# Fungsi untuk update pengguna
def update_user(username, new_password=None, new_role=None):
    users = load_users()
    users.loc[users['username'] == username, 'password'] = new_password if new_password else users['password']
    users.loc[users['username'] == username, 'role'] = new_role if new_role else users['role']
    users.to_csv("data/users.csv", index=False)

# Fungsi untuk menghapus pengguna
def delete_user(username):
    users = load_users()
    users = users[users['username'] != username]
    users.to_csv("data/users.csv", index=False)

# Fungsi untuk menampilkan CRUD user
def admin_panel():
    st.title("Admin Panel - CRUD User")

    # Form untuk menambah user baru
    st.subheader("Tambah User")
    new_username = st.text_input("Username Baru")
    new_password = st.text_input("Password", type="password", key="admin_password_input")
    confirm_password = st.text_input("Konfirmasi Password", type="password")
    new_role = st.selectbox("Pilih Role", ["user", "admin"])

    if st.button("Tambah User"):
        if new_password != confirm_password:
            st.warning("Password tidak cocok!")
        elif new_username in load_users()['username'].values:
            st.warning("Username sudah terdaftar!")
        else:
            save_user(new_username, new_password, new_role)
            st.success(f"User {new_username} berhasil ditambahkan!")

    # Form untuk memperbarui user
    st.subheader("Update User")
    update_username = st.selectbox("Pilih User untuk Diupdate", options=load_users()["username"].tolist())
    update_password = st.text_input("Password Baru (kosongkan jika tidak ingin mengubah)", type="password")
    update_role = st.selectbox("Role Baru", ["user", "admin"])

    if st.button("Update User"):
        update_user(update_username, new_password=update_password, new_role=update_role)
        st.success(f"User {update_username} berhasil diperbarui!")

    # Form untuk menghapus user
    st.subheader("Hapus User")
    delete_username = st.selectbox("Pilih User untuk Dihapus", options=load_users()["username"].tolist())

    if st.button("Hapus User"):
        delete_user(delete_username)
        st.success(f"User {delete_username} berhasil dihapus!")

    # Menampilkan daftar pengguna
    st.subheader("Daftar Pengguna")
    users = load_users()
    st.dataframe(users)

# Panggil admin_panel jika ingin menampilkan tampilan CRUD user
admin_panel()

