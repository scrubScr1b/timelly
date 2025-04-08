import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title("Market Dashboard")

# Load data
df = pd.read_excel("data/df_mz.xlsx")

# Format datetime
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.strftime("%b")
df["Month_Num"] = df["Date"].dt.month

# Pilihan bulan dalam urutan yang benar
month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Sidebar filter
all_years = sorted(df["Year"].unique())
all_months = month_order
all_brands = sorted(df["Brand"].unique())

selected_years = st.sidebar.multiselect("Pilih Tahun", options=all_years, default=all_years)
selected_months = st.sidebar.multiselect("Pilih Bulan", options=all_months, default=all_months)
selected_brands = st.sidebar.multiselect("Pilih Brand", options=all_brands, default=all_brands)

# Filter data
df_filtered = df[(df["Year"].isin(selected_years)) & 
                 (df["Month"].isin(selected_months)) & 
                 (df["Brand"].isin(selected_brands))]

# Data Preview
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

with kpi3:
    st.metric("Unique Customers", f"{unique_customers}")

# Data Pivot
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Top Customers")
    pivot = pd.pivot_table(df_filtered, index="Customers", values=["Qty", "Total_Sales"], aggfunc="sum")
    pivot = pivot.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    pivot = pivot.sort_values(by="Total Sales", ascending=False)
    st.dataframe(pivot.style.format("{:,.0f}"))

with col2:
    st.subheader("Top Brand")
    pivot = pd.pivot_table(df_filtered, index="Brand", values=["Qty", "Total_Sales"], aggfunc="sum")
    pivot = pivot.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    pivot = pivot.sort_values(by="Total Sales", ascending=False)
    st.dataframe(pivot.style.format("{:,.0f}"))

with col3:
    st.subheader("Sales per Year")
    pivot_year = pd.pivot_table(df_filtered, index="Year", values=["Qty", "Total_Sales"], aggfunc="sum")
    pivot_year = pivot_year.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    st.dataframe(pivot_year.style.format("{:,.0f}"))

with col4:
    st.subheader("Sales per Month")
    pivot_month = pd.pivot_table(df_filtered, index="Month", values=["Qty", "Total_Sales"], aggfunc="sum")
    pivot_month = pivot_month.reindex(month_order)
    pivot_month = pivot_month.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})
    st.dataframe(pivot_month.style.format("{:,.0f}"))

# ===================
# 📊 VISUALISASI BAR
# ===================
st.markdown("---")
st.subheader("📊 Sales Visualizations")

chart_option = st.selectbox("Pilih jenis data untuk visualisasi:", ["Total Sales", "Quantity", "Total Sales & Quantity"])

# Top 10 Customers
st.markdown("**Top 10 Customers**")
top_customers_df = df_filtered.groupby("Customers")[["Qty", "Total_Sales"]].sum().sort_values(by="Total_Sales", ascending=False).head(10).reset_index()

if chart_option == "Total Sales":
    chart_top = alt.Chart(top_customers_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Total_Sales:Q", title="Total Sales"),
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Total_Sales"])
elif chart_option == "Quantity":
    chart_top = alt.Chart(top_customers_df).mark_bar(color="#ff7f0e").encode(
        x=alt.X("Qty:Q", title="Quantity"),
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Qty"])
else:
    bar = alt.Chart(top_customers_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Total_Sales:Q"),
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Total_Sales"])
    line = alt.Chart(top_customers_df).mark_line(color="#ff7f0e", point=True).encode(
        x="Qty:Q",
        y=alt.Y("Customers:N", sort='-x'),
        tooltip=["Customers", "Qty"])
    chart_top = bar + line

st.altair_chart(chart_top.properties(height=300), use_container_width=True)

# Sales per Month
st.markdown("**Sales per Month**")
sales_month_df = df_filtered.groupby("Month")[["Qty", "Total_Sales"]].sum().reindex(month_order).reset_index()

if chart_option == "Total Sales":
    chart_month = alt.Chart(sales_month_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Month:N", sort=month_order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q", title="Total Sales"),
        tooltip=["Month", "Total_Sales"])
elif chart_option == "Quantity":
    chart_month = alt.Chart(sales_month_df).mark_bar(color="#ff7f0e").encode(
        x=alt.X("Month:N", sort=month_order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q", title="Quantity"),
        tooltip=["Month", "Qty"])
else:
    bar = alt.Chart(sales_month_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Month:N", sort=month_order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q"),
        tooltip=["Month", "Total_Sales"])
    line = alt.Chart(sales_month_df).mark_line(color="#ff7f0e", point=True).encode(
        x=alt.X("Month:N", sort=month_order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q"),
        tooltip=["Month", "Qty"])
    chart_month = bar + line

st.altair_chart(chart_month.properties(height=300), use_container_width=True)

# Sales per Year
st.markdown("**Sales per Year**")
sales_year_df = df_filtered.groupby("Year")[["Qty", "Total_Sales"]].sum().reset_index()

if chart_option == "Total Sales":
    chart_year = alt.Chart(sales_year_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q", title="Total Sales"),
        tooltip=["Year", "Total_Sales"])
elif chart_option == "Quantity":
    chart_year = alt.Chart(sales_year_df).mark_bar(color="#ff7f0e").encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q", title="Quantity"),
        tooltip=["Year", "Qty"])
else:
    bar = alt.Chart(sales_year_df).mark_bar(color="#1f77b4").encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Total_Sales:Q"),
        tooltip=["Year", "Total_Sales"])
    line = alt.Chart(sales_year_df).mark_line(color="#ff7f0e", point=True).encode(
        x=alt.X("Year:O", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Qty:Q"),
        tooltip=["Year", "Qty"])
    chart_year = bar + line

st.altair_chart(chart_year.properties(height=300), use_container_width=True)

# ===================
# 🔥 Annotated Heatmap
# ===================
st.markdown("---")
st.subheader("Heatmap Penjualan (Tahun vs Bulan)")

heatmap_metric = st.radio("Pilih jenis data:", ["Total Sales", "Quantity"], horizontal=True)

heatmap_data = df_filtered.groupby(["Month", "Year"])[["Qty", "Total_Sales"]].sum().reset_index()
heatmap_pivot = heatmap_data.pivot(index="Month", columns="Year", values="Total_Sales" if heatmap_metric == "Total Sales" else "Qty")
heatmap_pivot = heatmap_pivot.reindex(index=month_order)

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_pivot, annot=True, fmt=".0f", cmap="Blues", linewidths=.5, ax=ax)
ax.set_title(f"{heatmap_metric} per Bulan dan Tahun")
st.pyplot(fig)
