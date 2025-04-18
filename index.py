import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz
from datetime import datetime

st.set_page_config(page_title="Timelly", layout="wide")

# ------------------------ USER MANAGEMENT ------------------------ #

def load_users():
    if not os.path.exists("data/users.csv"):
        pd.DataFrame(columns=["username", "password", "role"]).to_csv("data/users.csv", index=False)
    return pd.read_csv("data/users.csv")

def save_user(username, password):
    users = load_users()
    new_user = pd.DataFrame([[username, password, "user"]], columns=["username", "password", "role"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv("data/users.csv", index=False)

def login_user(username, password):
    users = load_users()
    matched = users[(users['username'] == username) & (users['password'] == password)]
    if not matched.empty:
        return True, matched.iloc[0]["role"]
    return False, None

def log_activity(username, action, detail=""):
    os.makedirs("data", exist_ok=True)
    log_path = "data/activity_log.csv"
    
    # Set timezone ke Indonesia (WIB)
    wib = pytz.timezone("Asia/Jakarta")
    timestamp = datetime.now(wib).strftime("%Y-%m-%d %H:%M:%S")
    
    entry = pd.DataFrame([[timestamp, username, action, detail]],
                         columns=["timestamp", "username", "action", "detail"])
    
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        entry = pd.concat([df, entry], ignore_index=True)
    
    entry.to_csv(log_path, index=False)

# ------------------------ AUTH FORMS ------------------------ #

def login_form():
    st.subheader("Login Into Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        success, role = login_user(username, password)
        if success:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            log_activity(username, "login", "success")
            st.success("Login berhasil!")
            st.rerun()
        else:
            st.error("Username atau password salah.")
            log_activity(username, "login", "failed")

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
            log_activity(username, "register", "user registered")
            st.rerun()

def logout():
    if st.sidebar.button("Logout"):
        log_activity(st.session_state["username"], "logout")
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["role"] = ""
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

            user_match = (users["username"] == username) & (users["password"] == current_pw)
            if not user_match.any():
                st.error("Password lama salah.")
                return
            if new_pw != confirm_pw:
                st.warning("Password baru tidak cocok.")
                return

            users.loc[user_match, "password"] = new_pw
            users.to_csv("data/users.csv", index=False)
            log_activity(username, "change_password", "password updated")
            st.success("Password berhasil diubah.")

# ------------------------ SESSION INIT ------------------------ #

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""

# ------------------------ AUTH FLOW ------------------------ #

if not st.session_state["logged_in"]:
    st.title("Holla Welcome to Timelly!")
    auth_mode = st.radio("Select Action", ["Login", "Register"], horizontal=True)

    if auth_mode == "Login":
        login_form()
    else:
        register_form()

    st.stop()

# ------------------------ POST LOGIN VIEW ------------------------ #

st.sidebar.success(f"Halo, {st.session_state['username']} !")
logout()
change_password_modal()

# ------------------------ ROLE-BASED NAVIGATION ------------------------ #

def admin_panel():
    st.subheader("Admin Panel - Log Aktivitas")
    log_path = "data/activity_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        st.dataframe(df)
    else:
        st.info("Belum ada log aktivitas.")

user_pages = {
    "Dataset": [
        st.Page("app_pages/preview.py", title="Preview Dataset"),
    ],
    "Market": [
        st.Page("app_pages/dashboard.py", title="Dashboard"),
        st.Page("app_pages/heatmap.py", title="Heat Map"),
    ],
    "Data ETL": [
        st.Page("app_pages/cleaning.py", title="Data Cleaning"),
    ],
    "Forecasting": [
        st.Page("app_pages/forecast.py", title="SARIMAX x BLSTM"),
    ]
}

admin_pages = {
    **user_pages,
    "Admin Panel": [
        st.Page(admin_panel, title="Log Aktivitas"),
        st.Page("app_pages/crud.py", title="Manage User"),
        st.Page("app_pages/dataset.py", title="Upload Dataset"),
    ]
}

if st.session_state["logged_in"]:
    pages = admin_pages if st.session_state["role"] == "admin" else user_pages
    pg = st.navigation(pages)
    pg.run()

