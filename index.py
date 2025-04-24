import streamlit as st
import pandas as pd
import os
from datetime import datetime
import pytz
import time
from pages.sidebar import admin_pages, user_pages

st.set_page_config(page_title="timelly", layout="wide", initial_sidebar_state="collapsed")

# -------------------- Hide sidebar collapse control saat belum login --------------------
if not st.session_state.get("logged_in", False):
    st.markdown("""
        <style>
        [data-testid="collapsedControl"] {display: none}
        </style>
    """, unsafe_allow_html=True)

# ------------------------ SESSION INIT ------------------------ #
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.session_state["role"] = ""

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
    if st.session_state.get("just_registered"):
        st.success("Registration successful! Please log in.")
        st.session_state["just_registered"] = False  # Reset setelah ditampilkan

    st.subheader("Login Into Account")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    login_btn = st.button("Login", key="login_btn")

    if login_btn:
        success, role = login_user(username, password)
        if success:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            log_activity(username, "login", "success")
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Incorrect username or password.")
            log_activity(username, "login", "failed")

# Register V2 nambahin pass tidak boleh ada spasi
def register_form():
    st.subheader("Regsiter Akun")
    username = st.text_input("Username Baru", key="register_username")
    password = st.text_input("Password", type="password", key="register_password")
    confirm = st.text_input("Konfirmasi Password", type="password", key="register_confirm")
    register_btn = st.button("Daftar", key="register_btn")

    if register_btn:
        if not username or not password or not confirm:
            st.warning("Semua kolom harus diisi!")
        elif " " in username:
            st.warning("Username tidak boleh mengandung spasi!")
        elif " " in password:
            st.warning("Password tidak boleh mengandung spasi!")
        elif password != confirm:
            st.warning("Password dan konfirmasi tidak cocok!")
        elif username in load_users()['username'].values:
            st.warning("Username sudah terdaftar!")
        else:
            save_user(username, password)
            log_activity(username, "register", "user registered")
            st.session_state["just_registered"] = True
            st.session_state["auth_mode"] = "Login"  # otomatis pindah ke login
            st.rerun()

def logout():
    if st.sidebar.button("Logout"):
        log_activity(st.session_state["username"], "logout")
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["role"] = ""
        st.rerun()

def change_password_modal():
    with st.sidebar.popover("Change Password"):
        st.subheader("Change Password")
        current_pw = st.text_input("Current Password", type="password", key="old_pw")
        new_pw = st.text_input("New Password", type="password", key="new_pw")
        confirm_pw = st.text_input("Confirm New Password", type="password", key="confirm_pw")
        if st.button("Update Password", key="update_pw_btn"):
            users = load_users()
            username = st.session_state["username"]

            user_match = (users["username"] == username) & (users["password"] == current_pw)
            if not user_match.any():
                st.error("Incorrect current password.")
                return
            if new_pw != confirm_pw:
                st.warning("New passwords do not match.")
                return

            users.loc[user_match, "password"] = new_pw
            users.to_csv("data/users.csv", index=False)
            log_activity(username, "change_password", "password updated")
            st.success("Password successfully updated.")

#  ------------------------ WELCOME PAGE SEKILAS ------------------------ #
if "show_welcome" not in st.session_state:
    st.session_state["show_welcome"] = True

if not st.session_state["logged_in"] and st.session_state["show_welcome"]:
    # Menampilkan halaman penyambutan sebelum login
    st.title("~timelly")

    st.markdown("""
        timelly adalah platform analitik penjualan interaktif yang membantu Anda memantau dan memahami data penjualan dengan mudah dan efisien.

        #### Fitur :
        - Visualisasi data penjualan secara real-time
        - Analisis pelanggan dan produk terlaris
        - Log aktivitas pengguna
        - Sistem login dan manajemen user
        - Prediksi Penjualan

        ---
        Klik tombol di bawah untuk mulai menggunakan aplikasi
    """)

    if st.button("Mulai"):
        # Menyembunyikan welcome page setelah tombol di klik
        st.session_state["show_welcome"] = False
        st.rerun()  # Menyegarkan halaman agar welcome page hilang
    
    st.stop()  # Menghentikan eksekusi lebih lanjut di sini sehingga navigasi tidak berjalan

# ------------------------ AUTH FLOW ------------------------ #
# Menampilkan halaman login atau register jika pengguna belum login
if not st.session_state["logged_in"]:
    st.title("Welcome to timelly!")

    # simpan mode login/register dalam session
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "Login"

    # tampilkan radio dan update state dengan benar
    auth_mode = st.radio(
        "Select Action", ["Login", "Register"],
        horizontal=True,
        index=0 if st.session_state["auth_mode"] == "Login" else 1,
        key="auth_radio"
    )

    if auth_mode != st.session_state["auth_mode"]:
        st.session_state["auth_mode"] = auth_mode
        st.rerun()  # versi baru dari experimental_rerun()

    # tampilkan form sesuai mode
    if st.session_state["auth_mode"] == "Login":
        login_form()
    else:
        register_form()

    st.stop()  # Menghentikan eksekusi lebih lanjut di sini sehingga navigasi tidak berjalan

# ------------------------ POST LOGIN VIEW ------------------------ #
# Sisa kode setelah welcome page yang hanya akan berjalan setelah user login
st.sidebar.success(f"Hello, {st.session_state['username']}!")
logout()
change_password_modal()

pages = admin_pages if st.session_state["role"] == "admin" else user_pages
pg = st.navigation(pages)
pg.run()

