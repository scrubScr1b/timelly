# import streamlit as st
# import pandas as pd
# import os
# import time

# # Fungsi untuk load data pengguna
# def load_users():
#     if not os.path.exists("data/users.csv"):
#         pd.DataFrame(columns=["username", "password", "role"]).to_csv("data/users.csv", index=False)
#     return pd.read_csv("data/users.csv")

# # Fungsi untuk menyimpan pengguna baru
# def save_user(username, password, role="user"):
#     users = load_users()
#     new_user = pd.DataFrame([[username, password, role]], columns=["username", "password", "role"])
#     users = pd.concat([users, new_user], ignore_index=True)
#     users.to_csv("data/users.csv", index=False)

# # Fungsi untuk update pengguna
# def update_user(username, new_password=None, new_role=None):
#     users = load_users()
#     users.loc[users['username'] == username, 'password'] = new_password if new_password else users['password']
#     users.loc[users['username'] == username, 'role'] = new_role if new_role else users['role']
#     users.to_csv("data/users.csv", index=False)

# # Fungsi untuk menghapus pengguna
# def delete_user(username):
#     users = load_users()
#     users = users[users['username'] != username]
#     users.to_csv("data/users.csv", index=False)

# # Fungsi untuk menampilkan CRUD user
# def admin_panel():
#     st.title("Admin Panel - CRUD User")

#     # Form untuk menambah user baru
#     st.subheader("Tambah User")
#     new_username = st.text_input("Username Baru")
#     new_password = st.text_input("Password", type="password", key="admin_password_input")
#     confirm_password = st.text_input("Konfirmasi Password", type="password")
#     new_role = st.selectbox("Pilih Role", ["user", "admin"])

#     if st.button("Tambah User"):
#         if new_password != confirm_password:
#             st.warning("Password tidak cocok!")
#         elif new_username in load_users()['username'].values:
#             st.warning("Username sudah terdaftar!")
#         else:
#             save_user(new_username, new_password, new_role)
#             st.success(f"User {new_username} berhasil ditambahkan!")
#             time.sleep(2)
#             st.rerun()

#     # Form untuk memperbarui user
#     st.subheader("Update User")
#     update_username = st.selectbox("Pilih User untuk Diupdate", options=load_users()["username"].tolist())
#     update_password = st.text_input("Password Baru (kosongkan jika tidak ingin mengubah)", type="password")
#     update_role = st.selectbox("Role Baru", ["user", "admin"])

#     if st.button("Update User"):
#         update_user(update_username, new_password=update_password, new_role=update_role)
#         st.success(f"User {update_username} berhasil diperbarui!")
#         time.sleep(2)
#         st.rerun()

    

#     # Form untuk menghapus user
#     st.subheader("Hapus User")
#     delete_username = st.selectbox("Pilih User untuk Dihapus", options=load_users()["username"].tolist())

#     if st.button("Hapus User"):
#         delete_user(delete_username)
#         st.success(f"User {delete_username} berhasil dihapus!")
#         time.sleep(2)
#         st.rerun()

#     # Menampilkan daftar pengguna
#     st.subheader("Daftar Pengguna")
#     users = load_users()
#     st.dataframe(users)

# # Panggil admin_panel jika ingin menampilkan tampilan CRUD user
# admin_panel()

# V2
# import streamlit as st
# import pandas as pd
# import os
# import time

# # Fungsi load data
# def load_users():
#     if not os.path.exists("data/users.csv"):
#         pd.DataFrame(columns=["username", "password", "role"]).to_csv("data/users.csv", index=False)
#     return pd.read_csv("data/users.csv")

# # Fungsi simpan user baru
# def save_user(username, password, role="user"):
#     users = load_users()
#     new_user = pd.DataFrame([[username, password, role]], columns=["username", "password", "role"])
#     users = pd.concat([users, new_user], ignore_index=True)
#     users.to_csv("data/users.csv", index=False)

# # Fungsi update user
# def update_user(username, new_password=None, new_role=None):
#     users = load_users()
#     if new_password:
#         users.loc[users['username'] == username, 'password'] = new_password
#     if new_role:
#         users.loc[users['username'] == username, 'role'] = new_role
#     users.to_csv("data/users.csv", index=False)

# # Fungsi hapus user
# def delete_user(username):
#     users = load_users()
#     users = users[users['username'] != username]
#     users.to_csv("data/users.csv", index=False)

# # Fungsi utama
# def admin_panel():
#     st.title("Admin Panel - CRUD User")

#     # Tambah User
#     st.subheader("Tambah User")
#     with st.form("form_tambah", clear_on_submit=True):
#         new_username = st.text_input("Username Baru")
#         new_password = st.text_input("Password", type="password")
#         confirm_password = st.text_input("Konfirmasi Password", type="password")
#         new_role = st.selectbox("Pilih Role", ["user", "admin"])
#         submitted = st.form_submit_button("Tambah User")

#         if submitted:
#             if new_password != confirm_password:
#                 st.warning("Password tidak cocok!")
#             elif new_username in load_users()['username'].values:
#                 st.warning("Username sudah terdaftar!")
#             else:
#                 save_user(new_username, new_password, new_role)
#                 st.success(f"User {new_username} berhasil ditambahkan!")
#                 time.sleep(1)
#                 st.rerun()

#     # Update User
#     st.subheader("Update User")
#     users = load_users()
#     if not users.empty:
#         with st.form("form_update", clear_on_submit=True):
#             update_username = st.selectbox("Pilih User", users["username"].tolist())
#             update_password = st.text_input("Password Baru (kosongkan jika tidak diubah)", type="password")
#             update_role = st.selectbox("Role Baru", ["user", "admin"])
#             update_submit = st.form_submit_button("Update User")

#             if update_submit:
#                 update_user(update_username, update_password if update_password else None, update_role)
#                 st.success(f"User {update_username} berhasil diperbarui!")
#                 time.sleep(1)
#                 st.rerun()

#     # Hapus User
#     st.subheader("Hapus User")
#     if not users.empty:
#         with st.form("form_delete", clear_on_submit=True):
#             delete_username = st.selectbox("Pilih User untuk Dihapus", users["username"].tolist())
#             delete_submit = st.form_submit_button("Hapus User")

#             if delete_submit:
#                 delete_user(delete_username)
#                 st.success(f"User {delete_username} berhasil dihapus!")
#                 time.sleep(1)
#                 st.rerun()

#     # Tabel Data Pengguna
#     st.subheader("Daftar Pengguna")
#     st.dataframe(load_users())

# # Jalankan
# admin_panel()


# V3
import streamlit as st
import pandas as pd
import os
import time

# Fungsi load data
def load_users():
    if not os.path.exists("data/users.csv"):
        pd.DataFrame(columns=["username", "password", "role"]).to_csv("data/users.csv", index=False)
    return pd.read_csv("data/users.csv")

# Fungsi simpan user baru
def save_user(username, password, role="user"):
    users = load_users()
    new_user = pd.DataFrame([[username, password, role]], columns=["username", "password", "role"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("data/users.csv", index=False)

# Fungsi update user
def update_user(username, new_password=None, new_role=None):
    users = load_users()
    if new_password:
        users.loc[users['username'] == username, 'password'] = new_password
    if new_role:
        users.loc[users['username'] == username, 'role'] = new_role
    users.to_csv("data/users.csv", index=False)

# Fungsi hapus user
def delete_user(username):
    users = load_users()
    users = users[users['username'] != username]
    users.to_csv("data/users.csv", index=False)

# Fungsi utama
def admin_panel():
    st.title("Admin Panel - CRUD User")

    # Tambah User
    st.subheader("Tambah User")
    if "new_username" not in st.session_state:
        st.session_state.new_username = ""
        st.session_state.new_password = ""
        st.session_state.confirm_password = ""
        st.session_state.new_role = "user"

    with st.form("form_tambah", clear_on_submit=True):
        new_username = st.text_input("Username Baru", value=st.session_state.new_username)
        new_password = st.text_input("Password", type="password", value=st.session_state.new_password)
        confirm_password = st.text_input("Konfirmasi Password", type="password", value=st.session_state.confirm_password)
        new_role = st.selectbox("Pilih Role", ["user", "admin"], index=["user", "admin"].index(st.session_state.new_role))
        submitted = st.form_submit_button("Tambah User")

        if submitted:
            if new_password != confirm_password:
                st.warning("Password tidak cocok!")
            elif new_username in load_users()['username'].values:
                st.warning("Username sudah terdaftar!")
            else:
                save_user(new_username, new_password, new_role)
                st.success(f"User {new_username} berhasil ditambahkan!")
                # Update session state to reset form fields
                st.session_state.new_username = ""
                st.session_state.new_password = ""
                st.session_state.confirm_password = ""
                st.session_state.new_role = "user"

    # Update User
    st.subheader("Update User")
    users = load_users()
    if not users.empty:
        with st.form("form_update", clear_on_submit=True):
            update_username = st.selectbox("Pilih User", users["username"].tolist())
            update_password = st.text_input("Password Baru (kosongkan jika tidak diubah)", type="password")
            update_role = st.selectbox("Role Baru", ["user", "admin"])
            update_submit = st.form_submit_button("Update User")

            if update_submit:
                update_user(update_username, update_password if update_password else None, update_role)
                st.success(f"User {update_username} berhasil diperbarui!")

    # Hapus User
    st.subheader("Hapus User")
    if not users.empty:
        with st.form("form_delete", clear_on_submit=True):
            delete_username = st.selectbox("Pilih User untuk Dihapus", users["username"].tolist())
            delete_submit = st.form_submit_button("Hapus User")

            if delete_submit:
                delete_user(delete_username)
                st.success(f"User {delete_username} berhasil dihapus!")

    # Tabel Data Pengguna
    st.subheader("Daftar Pengguna")
    st.dataframe(load_users())

# Jalankan
admin_panel()
