import streamlit as st
import pandas as pd
import os

st.subheader("Admin Panel - Activity Logs")
log_path = "data/activity_log.csv"
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
    st.dataframe(df)
else:
    st.info("Belum ada log aktivitas.")