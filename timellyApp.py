import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Market Dashboard", layout="wide")

# Data Dummy untuk tabel
data = {
    "#": list(range(1, 11)),
    "Department": [f"DEPT-{i:02}" for i in range(1, 11)],
    "ID Customer": [f"CUST-000{i}" for i in range(1, 11)],
    "Customer": [f"Customer Name 00{i}" for i in range(1, 11)],
    "No Invoice": [f"IVC24-12{15 + i}" for i in range(1, 11)],
    "Date": [f"{15 + i}/02/2024" for i in range(1, 11)],
    "Code Product": ["CODE1005116"] * 10,
    "Description Product": ["Product Description 1"] * 10,
    "Brand": ["Brand 01"] * 10,
    "Category": ["Category 01"] * 10,
    "Retail": ["10.000"] * 10,
    "Qty": [10] * 10,
    "Total Sales": ["50.000"] * 10,
}

# Konversi ke DataFrame
df = pd.DataFrame(data)

# Header
st.title("Today's Market Dashboard")
st.write("The global net cap is $2.77T, a 0.04% increase over the last month")

# Layout Kolom
col1, col2, col3 = st.columns(3)

# Top Customer
with col1:
    st.subheader("Top Customer")
    top_customers = pd.DataFrame({"Customer": [f"Customer {i+1}" for i in range(5)], "Qty": [10]*5, "Total Sales": ["10.000.000"]*5})
    st.dataframe(top_customers, hide_index=True)

# Top Brand Product
with col2:
    st.subheader("Top Brand Product")
    top_products = pd.DataFrame({"Product": [f"Product {i+1}" for i in range(5)], "Qty": [10]*5, "Total Sales": ["10.000.000"]*5})
    st.dataframe(top_products, hide_index=True)

# Total Sales & Qty
with col3:
    st.metric(label="Total Sales", value="50.000.000")
    st.metric(label="Total Qty", value=50)

# Tabel Market Data Analysis
st.subheader("Market Data Analysis")
st.dataframe(df, hide_index=True)
