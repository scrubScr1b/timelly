import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Customer Analysis", layout="wide")
st.title("Customer Sales Analysis")

# Cek apakah data tersedia
if "data" not in st.session_state:
    st.warning("Silakan upload file terlebih dahulu di halaman utama.")
    st.stop()

df = st.session_state["data"]

# Pastikan kolom date dalam format datetime
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.strftime("%b")

# Urutan bulan
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Sidebar filters - multiple select
selected_years = st.sidebar.multiselect(
    "Pilih Tahun", sorted(df["year"].dropna().unique(), reverse=True),
    default=sorted(df["year"].dropna().unique(), reverse=True)
)

selected_months = st.sidebar.multiselect(
    "Pilih Bulan", month_order,
    default=month_order
)

# Filter data
df_filtered = df[df["year"].isin(selected_years) & df["month"].isin(selected_months)]

# 📊 Section 1: Heatmap - Monthly Sales per Customer
st.subheader("Heatmap - Monthly Sales per Customer")

# Pastikan urutan bulan
df_filtered["month"] = pd.Categorical(df_filtered["month"], categories=month_order, ordered=True)

# Pivot table untuk heatmap
heatmap_data = df_filtered.pivot_table(
    index="customers", columns="month", values="total_sales", aggfunc="sum", fill_value=0
)
heatmap_data = heatmap_data[month_order]  # Urutkan kolom bulan

# Buat heatmap
fig_heatmap = px.imshow(
    heatmap_data,
    color_continuous_scale="Viridis",
    labels=dict(x="Month", y="Customer", color="Total Sales"),
    aspect="auto",
    title=f"Monthly Sales per Customer ({', '.join(map(str, selected_years))})"
)

fig_heatmap.update_layout(
    xaxis_side="top",
    margin=dict(l=0, r=0, b=0, t=50),
    coloraxis_colorbar=dict(title="Sales", tickformat=","),
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# 📊 Section 2: Heatmap - Top N Customer
st.subheader("Heatmap - Top N Customer")

# 🎯 SLIDER Top N Customer DIPINDAHKAN ke sini (bukan di sidebar)
top_n = st.slider("Tampilkan Top N Customer berdasarkan Total Sales", min_value=5, max_value=50, value=20)

# Hitung total sales per customer dan ambil Top N
top_customers = (
    df_filtered.groupby("customers")["total_sales"]
    .sum()
    .nlargest(top_n)
    .index
)

# Filter data hanya untuk top N customer
df_top = df_filtered[df_filtered["customers"].isin(top_customers)]

# Urutkan bulan
df_top["month"] = pd.Categorical(df_top["month"], categories=month_order, ordered=True)

# Buat pivot untuk heatmap
heatmap_data = df_top.pivot_table(
    index="customers", columns="month", values="total_sales", aggfunc="sum", fill_value=0
)
heatmap_data = heatmap_data[month_order]

# Tampilkan heatmap
fig_heatmap = px.imshow(
    heatmap_data,
    color_continuous_scale="Viridis",
    labels=dict(x="Month", y="Customer", color="Total Sales"),
    aspect="auto",
    title=f"Heatmap Penjualan Bulanan Top {top_n} Customer - {', '.join(map(str, selected_years))}"
)

fig_heatmap.update_layout(
    xaxis_side="top",
    margin=dict(l=0, r=0, b=0, t=50),
    coloraxis_colorbar=dict(title="Sales", tickformat=","),
)

st.plotly_chart(fig_heatmap, use_container_width=True)
