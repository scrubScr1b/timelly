import streamlit as st
import pandas as pd


st.title("Data Preview")

@st.cache_data
def load_data(file):
    data = pd.read_excel(file)
    return data

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is None:
    st.info("Upload a file through config")
    st.stop()

df = load_data(uploaded_file)
st.dataframe(df)
