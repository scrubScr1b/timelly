import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="Market Dashboard", layout="wide")

# Sidebar Navigation dengan ikon dan struktur seperti gambar
st.sidebar.markdown(
    """
    <style>
        .sidebar-section {
            padding: 10px;
            font-size: 16px;
            font-weight: bold;
            color: #6c757d;
        }
        .sidebar-link {
            padding: 8px;
            text-decoration: none;
            display: flex;
            align-items: center;
            color: black;
            font-weight: normal;
        }
        .sidebar-link:hover {
            background-color: #f0f0f0;
            font-weight: bold;
        }
        .icon {
            margin-right: 10px;
        }
    </style>
    <div class="sidebar-section">Account</div>
    <a href="?page=Logout" class="sidebar-link">🔓 Log out</a>
    <div class="sidebar-section">Reports</div>
    <a href="?page=Dashboard" class="sidebar-link">📊 Dashboard</a>
    <a href="?page=BugReports" class="sidebar-link">⚙️ Bug reports</a>
    <a href="?page=SystemAlerts" class="sidebar-link">🔔 System alerts</a>
    <div class="sidebar-section">Tools</div>
    <a href="?page=Search" class="sidebar-link">🔍 Search</a>
    <a href="?page=History" class="sidebar-link">⏳ History</a>
    """,
    unsafe_allow_html=True
)

# Navbar
st.markdown(
    """
    <style>
        .navbar {
            background-color: #ffffff;
            padding: 10px;
            display: flex;
            align-items: center;
            font-size: 18px;
            font-weight: bold;
        }
        .navbar a {
            margin-right: 20px;
            text-decoration: none;
            color: black;
            font-weight: normal;
        }
        .navbar a:hover {
            font-weight: bold;
        }
    </style>
    <div class="navbar">
        <span><strong>((Logo)) Timelly</strong></span>
        <a href="?page=Market Dashboard">Market</a>
        <a href="?page=Forecasting">Forecasting</a>
        <a href="?page=Dataset">Dataset</a>
    </div>
    """,
    unsafe_allow_html=True
)

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
