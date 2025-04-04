import pandas as pd
import streamlit as st

st.title("Market Dashboard")

# Load data
df = pd.read_excel("data/df_mz.xlsx")

# Pastikan kolom Date dalam format datetime
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year  # Ambil ta`hun
df["Month"] = df["Date"].dt.strftime("%b")  # Ambil bulan dalam format singkat (Jan, Feb, dst.)

# Sidebar untuk filter Tahun
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df["Year"].unique(), reverse=True))

# Sidebar untuk filter Brand (bisa memilih lebih dari satu)
selected_brands = st.sidebar.multiselect("Pilih Brand", df["Brand"].unique(), default=df["Brand"].unique())

# Filter data berdasarkan pilihan user
df_filtered = df[(df["Year"] == selected_year) & (df["Brand"].isin(selected_brands))]

# Menampilkan Data yang sudah difilter
with st.expander("Data Preview"):
    st.dataframe(df_filtered)

col1, col2, col3, col4 = st.columns(4)

# Market Dashboard (Customers)
with col1:
    st.subheader("Top Customers")
    pivot = pd.pivot_table(
        df_filtered,
        index="Customers",
        values=["Qty", "Total_Sales"],
        aggfunc="sum",
    ).sort_values(by="Total_Sales", ascending=False)

    pivot = pivot.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    st.dataframe(pivot.style.format("{:,.0f}"))

# Top Brand
with col2:
    st.subheader("Top Brand")

    pivot = pd.pivot_table(
        df_filtered,
        index="Brand",
        values=["Qty", "Total_Sales"],
        aggfunc="sum",
    ).sort_values(by="Total_Sales", ascending=False)

    pivot = pivot.rename(columns={"Qty": "Quantity", "Total Sales": "Total Sales"})
    st.dataframe(pivot.style.format("{:,.0f}"))

# Sales per Year
with col3:
    st.subheader("Sales per Year")

    pivot_year = pd.pivot_table(
        df,
        index="Year",
        values=["Qty", "Total_Sales"],
        aggfunc="sum"
    ).sort_index(ascending=True)  # Urutkan tahun dari yang terlama ke terbaru

    pivot_year = pivot_year.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    st.dataframe(pivot_year.style.format("{:,.0f}"))

# Sales per Month
with col4:
    st.subheader("Sales per Month")

    pivot_month = pd.pivot_table(
        df_filtered,
        index="Month",
        values=["Qty", "Total_Sales"],
        aggfunc="sum"
    ).reindex(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    pivot_month = pivot_month.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    st.dataframe(pivot_month.style.format("{:,.0f}"))