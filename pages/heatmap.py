import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Analysis", layout="wide")
st.title("📊 Customer Sales Analysis")

# Load data
df = pd.read_excel("data/df_mz.xlsx")

# Pastikan kolom Date dalam format datetime
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.strftime("%b")

# Sidebar filters
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df["Year"].dropna().unique(), reverse=True))
selected_month = st.sidebar.selectbox("Pilih Bulan", ["All"] + list(df["Month"].dropna().unique()))

# Filter data
df_filtered = df[df["Year"] == selected_year]
if selected_month != "All":
    df_filtered = df_filtered[df_filtered["Month"] == selected_month]

# **📌 Section 1: Treemap Chart (Sales Contribution per Customer)**
st.subheader("📊 Treemap - Sales Contribution per Customer")

# Agregasi data berdasarkan pelanggan
sales_data = df_filtered.groupby("Customers")["Total_Sales"].sum().reset_index()

# Hitung total sales & persentase kontribusi
total_sales = sales_data["Total_Sales"].sum()
sales_data["Sales_Percentage"] = (sales_data["Total_Sales"] / total_sales) * 100

# Gabungkan nama customer + persen ke dalam satu label
sales_data["Label"] = (
    sales_data["Customers"] + "<br>" +
    sales_data["Sales_Percentage"].round(2).astype(str) + "%"
)

# Buat Treemap
fig_treemap = px.treemap(
    sales_data,
    path=["Label"],
    values="Total_Sales",
    title=f"Total Sales Contribution per Customer ({selected_year}, {selected_month if selected_month != 'All' else 'All Months'})",
    color="Total_Sales",
    color_continuous_scale="greens",
    hover_data={"Total_Sales": ":,.0f", "Sales_Percentage": ":.2f"}
)

# Menampilkan label persen tepat di tengah
fig_treemap.update_traces(textinfo="label+value", texttemplate="%{label}<br>%{value:,}", textposition="middle center")

# Tampilkan Treemap di Streamlit
st.plotly_chart(fig_treemap, use_container_width=True)

# **📌 Section 2: Scatter Plot - Sales vs. Transaction Count**
st.subheader("📈 Scatter Plot - Sales vs. Transaction Count")

# Agregasi jumlah transaksi per customer
transaction_count = df_filtered.groupby("Customers")["Total_Sales"].count().reset_index()
transaction_count.columns = ["Customers", "Transaction_Count"]

# Gabungkan dengan total sales
scatter_data = sales_data.merge(transaction_count, on="Customers")

# Buat Scatter Plot
fig_scatter = px.scatter(
    scatter_data, 
    x="Transaction_Count", 
    y="Total_Sales", 
    size="Total_Sales", 
    color="Total_Sales",
    title="Total Sales vs. Transaction Count per Customer",
    hover_data=["Customers", "Transaction_Count"],
    color_continuous_scale="Blues"
)

# Tampilkan Scatter Plot di Streamlit
st.plotly_chart(fig_scatter, use_container_width=True)
