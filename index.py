# import streamlit as st
# import pandas as pd
# import os
# from datetime import datetime
# import pytz
# from pages.sidebar import admin_pages, user_pages
# import time

# st.set_page_config(page_title="timelly", layout="wide",initial_sidebar_state="collapsed")

# # Tambahkan CSS untuk sembunyikan sidebar tombol:
# if not st.session_state.get("logged_in", False):
#     hide_sidebar = """
#         <style>
#         [data-testid="collapsedControl"] {display: none}
#         </style>
#     """
#     st.markdown(hide_sidebar, unsafe_allow_html=True)

# # ------------------------ SESSION INIT ------------------------ #

# if "logged_in" not in st.session_state:
#     st.session_state["logged_in"] = False
#     st.session_state["username"] = ""
#     st.session_state["role"] = ""

# # ------------------------ USER MANAGEMENT ------------------------ #

# def load_users():
#     if not os.path.exists("data/users.csv"):
#         pd.DataFrame(columns=["username", "password", "role"]).to_csv("data/users.csv", index=False)
#     return pd.read_csv("data/users.csv")

# def save_user(username, password):
#     users = load_users()
#     new_user = pd.DataFrame([[username, password, "user"]], columns=["username", "password", "role"])
#     users = pd.concat([users, new_user], ignore_index=True)
#     users.to_csv("data/users.csv", index=False)

# def login_user(username, password):
#     users = load_users()
#     matched = users[(users['username'] == username) & (users['password'] == password)]
#     if not matched.empty:
#         return True, matched.iloc[0]["role"]
#     return False, None

# def log_activity(username, action, detail=""):
#     os.makedirs("data", exist_ok=True)
#     log_path = "data/activity_log.csv"

#     wib = pytz.timezone("Asia/Jakarta")
#     timestamp = datetime.now(wib).strftime("%Y-%m-%d %H:%M:%S")

#     entry = pd.DataFrame([[timestamp, username, action, detail]],
#                          columns=["timestamp", "username", "action", "detail"])

#     if os.path.exists(log_path):
#         df = pd.read_csv(log_path)
#         entry = pd.concat([df, entry], ignore_index=True)

#     entry.to_csv(log_path, index=False)

# # ------------------------ AUTH FORMS ------------------------ #


# def login_form():
#     st.subheader("Login Into Account")
#     username = st.text_input("Username")
#     password = st.text_input("Password", type="password")
#     login_btn = st.button("Login")

#     if login_btn:
#         success, role = login_user(username, password)
#         if success:
#             st.session_state["logged_in"] = True
#             st.session_state["username"] = username
#             st.session_state["role"] = role
#             log_activity(username, "login", "success")
#             st.success("Login successful!")
#             st.rerun()
#         else:
#             st.error("Incorrect username or password.")
#             log_activity(username, "login", "failed")

# def register_form():
#     st.subheader("Register New Account")
#     username = st.text_input("New Username")
#     password = st.text_input("Password", type="password")
#     confirm = st.text_input("Confirm Password", type="password")
#     register_btn = st.button("Register")

#     if register_btn:
#         if password != confirm:
#             st.warning("Passwords do not match!")
#         elif username in load_users()['username'].values:
#             st.warning("Username already registered!")
#         else:
#             save_user(username, password)
#             st.success("Registration successful! Please log in.")
#             log_activity(username, "register", "user registered")
#             time.sleep(2)
#             st.rerun()

# def logout():
#     if st.sidebar.button("Logout"):
#         log_activity(st.session_state["username"], "logout")
#         st.session_state["logged_in"] = False
#         st.session_state["username"] = ""
#         st.session_state["role"] = ""
#         st.rerun()

# def change_password_modal():
#     with st.sidebar.popover("Change Password"):
#         st.subheader("Change Password")
#         current_pw = st.text_input("Current Password", type="password", key="old_pw")
#         new_pw = st.text_input("New Password", type="password", key="new_pw")
#         confirm_pw = st.text_input("Confirm New Password", type="password", key="confirm_pw")
#         if st.button("Update Password", key="update_pw_btn"):
#             users = load_users()
#             username = st.session_state["username"]

#             user_match = (users["username"] == username) & (users["password"] == current_pw)
#             if not user_match.any():
#                 st.error("Incorrect current password.")
#                 return
#             if new_pw != confirm_pw:
#                 st.warning("New passwords do not match.")
#                 return

#             users.loc[user_match, "password"] = new_pw
#             users.to_csv("data/users.csv", index=False)
#             log_activity(username, "change_password", "password updated")
#             st.success("Password successfully updated.")

# # ------------------------ AUTH FLOW ------------------------ #

# if not st.session_state["logged_in"]:
#     st.title("Welcome to Timelly!")
#     auth_mode = st.radio("Select Action", ["Login", "Register"], horizontal=True)

#     if auth_mode == "Login":
#         login_form()
#     else:
#         register_form()

#     st.stop()

# # ------------------------ POST LOGIN VIEW ------------------------ #

# st.sidebar.success(f"Hello, {st.session_state['username']}!")
# logout()
# change_password_modal()

# pages = admin_pages if st.session_state["role"] == "admin" else user_pages
# pg = st.navigation(pages)
# pg.run()


# V2
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

def register_form():
    st.subheader("Register New Account")
    username = st.text_input("New Username", key="register_username")
    password = st.text_input("Password", type="password", key="register_password")
    confirm = st.text_input("Confirm Password", type="password", key="register_confirm")
    register_btn = st.button("Register", key="register_btn")

    if register_btn:
        if password != confirm:
            st.warning("Passwords do not match!")
        elif username in load_users()['username'].values:
            st.warning("Username already registered!")
        else:
            save_user(username, password)
            # st.success("Registration successful! Please log in.")
            log_activity(username, "register", "user registered")
            # time.sleep(1)
            st.session_state["just_registered"] = True #new
            st.session_state["auth_mode"] = "Login"  # switch otomatis ke login
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

# ------------------------ AUTH FLOW ------------------------ #

# V1 mentok di radio
# if not st.session_state["logged_in"]:
#     st.title("Welcome to Timelly!")

#     # simpan mode login/register dalam session
#     if "auth_mode" not in st.session_state:
#         st.session_state["auth_mode"] = "Login"

#     st.session_state["auth_mode"] = st.radio(
#         "Select Action", ["Login", "Register"],
#         horizontal=True,
#         index=0 if st.session_state["auth_mode"] == "Login" else 1
#     )

#     if st.session_state["auth_mode"] == "Login":
#         login_form()
#     else:
#         register_form()

#     st.stop()

# V2 
if not st.session_state["logged_in"]:
    st.title("Welcome to Timelly!")

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

    st.stop()


# ------------------------ POST LOGIN VIEW ------------------------ #

st.sidebar.success(f"Hello, {st.session_state['username']}!")
logout()
change_password_modal()

pages = admin_pages if st.session_state["role"] == "admin" else user_pages
pg = st.navigation(pages)
pg.run()
