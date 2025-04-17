import streamlit as st
import pandas as pd
import hashlib
import os

st.set_page_config(page_title="Timelly", layout="wide")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if not os.path.exists("users.csv"):
        pd.DataFrame(columns=["username", "password"]).to_csv("users.csv", index=False)
    return pd.read_csv("users.csv")

def save_user(username, password):
    users = load_users()
    new_user = pd.DataFrame([[username, hash_password(password)]], columns=["username", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("users.csv", index=False)

def login_user(username, password):
    users = load_users()
    hashed = hash_password(password)
    return ((users['username'] == username) & (users['password'] == hashed)).any()

def login_form():
    st.subheader("Login Into Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if login_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.success("Login berhasil!")
            st.rerun()
        else:
            st.error("Username atau password salah.")

def register_form():
    st.subheader("Register New Account")
    username = st.text_input("Username Baru")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Konfirmasi Password", type="password")
    register_btn = st.button("Register")

    if register_btn:
        if password != confirm:
            st.warning("Password tidak cocok!")
        elif username in load_users()['username'].values:
            st.warning("Username sudah terdaftar!")
        else:
            save_user(username, password)
            st.success("Registrasi berhasil! Silakan login.")
            st.rerun()

def logout():
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.rerun()

def change_password_modal():
    with st.sidebar.popover("Ganti Password"):
        st.subheader("Ganti Password")
        current_pw = st.text_input("Password Lama", type="password", key="old_pw")
        new_pw = st.text_input("Password Baru", type="password", key="new_pw")
        confirm_pw = st.text_input("Konfirmasi Password Baru", type="password", key="confirm_pw")
        if st.button("Update Password", key="update_pw_btn"):
            users = load_users()
            username = st.session_state["username"]
            hashed_current = hash_password(current_pw)

            user_match = (users["username"] == username) & (users["password"] == hashed_current)
            if not user_match.any():
                st.error("Password lama salah.")
                return
            if new_pw != confirm_pw:
                st.warning("Password baru tidak cocok.")
                return

            users.loc[user_match, "password"] = hash_password(new_pw)
            users.to_csv("users.csv", index=False)
            st.success("Password berhasil diubah.")


# Status awal
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Jika belum login
if not st.session_state["logged_in"]:
    st.title("Holla Welcome to Timelly!")
    auth_mode = st.radio("Select Action", ["Login", "Register"], horizontal=True)

    if auth_mode == "Login":
        login_form()
    else:
        register_form()

    st.stop()

# Setelah login
st.sidebar.success(f"Halo, {st.session_state['username']}!")
logout()
change_password_modal()


# Navigasi utama
pages = {
    "Dataset": [
        st.Page("pages/dataset.py", title="Upload Dataset"),
        st.Page("pages/preview.py", title="Preview Dataset"),
    ],
    "Market": [
        st.Page("pages/dashboard.py", title="Dashboard"),
        st.Page("pages/heatmap.py", title="Heat Map"),
    ],
    "Data ETL": [
        st.Page("pages/cleaning.py", title="Data Cleaning"),
    ],
    "Forecasting": [
        st.Page("pages/forecast.py", title="Coming Soon"),
    ],
}

pg = st.navigation(pages)
pg.run()
