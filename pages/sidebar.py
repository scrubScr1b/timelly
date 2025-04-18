# sidebar.py
import streamlit as st
import pandas as pd
import os

# ------------------------ ROLE-BASED NAVIGATION ------------------------ #

def admin_panel():
    st.subheader("Admin Panel - Activity Logs")
    log_path = "data/activity_log.csv"
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        st.dataframe(df)
    else:
        st.info("No activity logs yet.")

user_pages = {
    "Dataset": [
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
        st.Page("pages/forecast.py", title="SARIMAX x BLSTM"),
    ]
}

admin_pages = {
    **user_pages,
    "Admin Panel": [
        st.Page(admin_panel, title="Activity Logs"),
        st.Page("pages/crud.py", title="Manage Users"),
        st.Page("pages/dataset.py", title="Upload Dataset"),
    ]
}
