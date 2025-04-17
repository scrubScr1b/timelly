# import pandas as pd
# import streamlit as st

# st.title("Upload Dataset")

# Upload file
# uploaded_file = st.file_uploader("Upload Excel/CSV File", type=["xlsx", "csv"])

# if uploaded_file:
#     if uploaded_file.name.endswith(".csv"):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)

#     df.columns = df.columns.str.strip().str.lower()
#     st.session_state["data"] = df
#     st.success("Dataset berhasil dimuat!")

#     # Tampilkan nama kolom
#     st.write("Kolom yang terbaca:", df.columns.tolist())


import pandas as pd
import streamlit as st

st.title("Upload Dataset")

def upload_dataset():
    uploaded_file = st.file_uploader("Upload file CSV atau Excel", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.strip().str.lower()
        st.session_state["data"] = df
        st.success("Dataset berhasil dimuat!")
        st.write("Kolom yang terbaca:", st.session_state["data"].columns.tolist())

# Cek dan tampilkan jika dataset sudah tersedia
if "data" in st.session_state:
    st.write("Kolom yang terbaca:", st.session_state["data"].columns.tolist())
else:
    upload_dataset()


