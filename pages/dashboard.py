import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(layout="wide")

st.title("Market Dashboard")

# Load data
df = pd.read_excel("data/df_mz.xlsx")

# Pastikan kolom Date dalam format datetime
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

df["Month"] = df["Date"].dt.strftime("%b")  # Format singkat (Jan, Feb, dst.)
all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Sidebar untuk filter
selected_years = st.sidebar.multiselect("Pilih Tahun", sorted(df["Year"].unique(), reverse=True), default=sorted(df["Year"].unique(), reverse=True))
selected_months = st.sidebar.multiselect("Pilih Bulan", all_months, default=all_months)
selected_brands = st.sidebar.multiselect("Pilih Brand", sorted(df["Brand"].unique()), default=sorted(df["Brand"].unique()))

# Filter data berdasarkan pilihan user
df_filtered = df[
    (df["Year"].isin(selected_years)) &
    (df["Month"].isin(selected_months)) &
    (df["Brand"].isin(selected_brands))
]

# Menampilkan Data yang sudah difilter
with st.expander("Data Preview"):
    st.dataframe(df_filtered)

# KPI Section
total_sales = df_filtered["Total_Sales"].sum()
total_qty = df_filtered["Qty"].sum()
unique_customers = df_filtered["Customers"].nunique()

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric("Total Sales All Time", f"Rp {total_sales:,.0f}")

with kpi2:
    st.metric("Total Quantity All Time", f"{total_qty:,.0f}")

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

    pivot = pivot.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    st.dataframe(pivot.style.format("{:,.0f}"))

# Sales per Year
with col3:
    st.subheader("Sales per Year")

    pivot_year = pd.pivot_table(
        df_filtered,
        index="Year",
        values=["Qty", "Total_Sales"],
        aggfunc="sum"
    ).sort_index(ascending=True)

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
    ).reindex(all_months)

    pivot_month = pivot_month.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    st.dataframe(pivot_month.style.format("{:,.0f}"))

# =====================
# 📊 BAR CHART SECTION
# =====================

st.markdown("---")
st.subheader("📊 Sales Visualizations")

# Dropdown selector untuk jenis data
chart_option = st.selectbox(
    "Pilih jenis data untuk visualisasi:",
    ["Total Sales", "Quantity", "Total Sales & Quantity"]
)

# 1. Bar Chart - Top 10 Customers
st.markdown("**Top 10 Customers**")
top_customers_df = df_filtered.groupby("Customers")[["Qty", "Total_Sales"]].sum().sort_values(
    by="Total_Sales", ascending=False
).head(10).reset_index()

if chart_option == "Total Sales":
    chart_top = alt.Chart(top_customers_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Total_Sales:Q", title="Total Sales"),
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Total_Sales"]
    )
elif chart_option == "Quantity":
    chart_top = alt.Chart(top_customers_df).mark_bar(color="#ff7f0e").encode(
        x=alt.X("Qty:Q", title="Quantity"),
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Qty"]
    )
else:
    bar = alt.Chart(top_customers_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Total_Sales:Q"),
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Total_Sales"]
    )
    line = alt.Chart(top_customers_df).mark_line(color="#ff7f0e", point=True).encode(
        x="Qty:Q",
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Qty"]
    )
    chart_top = bar + line

st.altair_chart(chart_top.properties(height=300), use_container_width=True)

# 2. Bar Chart - Sales per Month
st.markdown("**Sales per Month**")
sales_month_df = df_filtered.groupby("Month")[["Qty", "Total_Sales"]].sum().reindex(
    all_months
).reset_index()

if chart_option == "Total Sales":
    chart_month = alt.Chart(sales_month_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Month:N", sort=list(sales_month_df["Month"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q", title="Total Sales"),
        tooltip=["Month", "Total_Sales"]
    )
elif chart_option == "Quantity":
    chart_month = alt.Chart(sales_month_df).mark_bar(color="#ff7f0e").encode(
        x=alt.X("Month:N", sort=list(sales_month_df["Month"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q", title="Quantity"),
        tooltip=["Month", "Qty"]
    )
else:
    bar = alt.Chart(sales_month_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Month:N", sort=list(sales_month_df["Month"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q"),
        tooltip=["Month", "Total_Sales"]
    )
    line = alt.Chart(sales_month_df).mark_line(color="#ff7f0e", point=True).encode(
        x=alt.X("Month:N", sort=list(sales_month_df["Month"]), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q"),
        tooltip=["Month", "Qty"]
    )
    chart_month = bar + line

st.altair_chart(chart_month.properties(height=300), use_container_width=True)

# 3. Bar Chart - Sales per Year
st.markdown("**Sales per Year**")
sales_year_df = df_filtered.groupby("Year")[["Qty", "Total_Sales"]].sum().reset_index()

if chart_option == "Total Sales":
    chart_year = alt.Chart(sales_year_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q", title="Total Sales"),
        tooltip=["Year", "Total_Sales"]
    )
elif chart_option == "Quantity":
    chart_year = alt.Chart(sales_year_df).mark_bar(color="#ff7f0e").encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q", title="Quantity"),
        tooltip=["Year", "Qty"]
    )
else:
    bar = alt.Chart(sales_year_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q"),
        tooltip=["Year", "Total_Sales"]
    )
    line = alt.Chart(sales_year_df).mark_line(color="#ff7f0e", point=True).encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q"),
        tooltip=["Year", "Qty"]
    )
    chart_year = bar + line

st.altair_chart(chart_year.properties(height=300), use_container_width=True)
