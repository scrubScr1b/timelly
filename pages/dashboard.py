import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from utils import load_saved_dataset 

st.title("Market Dashboard")

# Cek apakah data tersedia, jika belum coba load dari file
if "data" not in st.session_state:
    df, source = load_saved_dataset()
    if df is not None:
        st.session_state["data"] = df
        st.session_state["source"] = source
    else:
        if st.session_state.get("role") == "admin":
            st.warning("Silakan upload file terlebih dahulu di halaman admin")
        else:
            st.warning("Admin Belum Mengupload File Dataset!")
        st.stop()

df = st.session_state["data"]

# Cek dan format kolom date
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.strftime("%b")
    df["month_num"] = df["date"].dt.month
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
else:
    st.error("Kolom 'date' tidak ditemukan di file.")
    st.stop()

# Sidebar filters
all_years = sorted(df["year"].unique())
all_months = month_order
selected_years = st.sidebar.multiselect("Pilih Tahun", options=all_years, default=all_years)
selected_months = st.sidebar.multiselect("Pilih Bulan", options=all_months, default=all_months)

df_filtered = df[df["year"].isin(selected_years) & df["month"].isin(selected_months)]


#=============
# FILTER PANE
#=============

filter_values = {}
desired_order = ["dept", "customers", "brand"]
filter_candidates = [col for col in desired_order if col in df.columns]

for col in filter_candidates:
    if col in df_filtered.columns:
        temp_df = df_filtered.copy()
        for fcol, fval in filter_values.items():
            temp_df = temp_df[temp_df[fcol].isin(fval)]
        options = temp_df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(f"Pilih {col.title()}", options=options, default=options)
        filter_values[col] = selected

for fcol, fval in filter_values.items():
    df_filtered = df_filtered[df_filtered[fcol].isin(fval)]



# KPI
kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric("Total Sales", f"Rp {df_filtered['total_sales'].sum():,.0f}" if "total_sales" in df_filtered.columns else "N/A")
with kpi2:
    st.metric("Total Quantity", f"{df_filtered['qty'].sum():,.0f}" if "qty" in df_filtered.columns else "N/A")
with kpi3:
    st.metric("Unique Customers", f"{df_filtered['customers'].nunique()}" if "customers" in df_filtered.columns else "N/A")

# Pivot Tables
col1, col2, col3, col4 = st.columns(4)

# ====================
# 📊 Perbandingan Qty & Sales per Tahun per Brand
# ====================


st.markdown("---")
st.subheader("Perbandingan Total Sales per Tahun per Brand")

if "brand" in df_filtered.columns and "total_sales" in df_filtered.columns:

    # Agregasi total sales per brand per tahun
    combined = df_filtered.groupby(["brand", "year"]).agg({
        "total_sales": "sum"
    }).reset_index()

    # Pivot data ke bentuk tabel lebar
    pivot_sales = combined.pivot(index="brand", columns="year", values="total_sales").add_prefix("Sales_")

    # Hitung % perubahan antar tahun
    years = sorted(df_filtered["year"].unique())
    for i in range(1, len(years)):
        prev = years[i - 1]
        curr = years[i]
        col_prev = f"Sales_{prev}"
        col_curr = f"Sales_{curr}"
        delta_col = f"Δ_{curr}_vs_{prev}"

        if col_prev in pivot_sales.columns and col_curr in pivot_sales.columns:
            pivot_sales[delta_col] = ((pivot_sales[col_curr] - pivot_sales[col_prev]) / pivot_sales[col_prev]) * 100

    # Susun ulang kolom agar urut
    ordered_cols = []
    for y in years:
        if f"Sales_{y}" in pivot_sales.columns:
            ordered_cols.append(f"Sales_{y}")
        if y > years[0]:
            delta_col = f"Δ_{y}_vs_{years[years.index(y) - 1]}"
            if delta_col in pivot_sales.columns:
                ordered_cols.append(delta_col)

    full_table = pivot_sales[ordered_cols].reset_index()

    # Tambahkan baris TOTAL
    total_row = {"brand": "TOTAL"}
    for col in full_table.columns:
        if col.startswith("Sales_"):
            total_row[col] = full_table[col].sum()

    # Hitung Δ_% total antar tahun
    for i in range(1, len(years)):
        prev = years[i - 1]
        curr = years[i]
        col_prev = f"Sales_{prev}"
        col_curr = f"Sales_{curr}"
        delta_col = f"Δ_{curr}_vs_{prev}"

        if col_prev in full_table.columns and col_curr in full_table.columns:
            total_prev = full_table[col_prev].sum()
            total_curr = full_table[col_curr].sum()

            if total_prev != 0:
                total_row[delta_col] = ((total_curr - total_prev) / total_prev) * 100
            else:
                total_row[delta_col] = None

    # Sisipkan baris total ke bawah
    full_table.loc[len(full_table)] = total_row

    # Format styling
    format_dict = {}
    for col in full_table.columns:
        if col.startswith("Sales_"):
            format_dict[col] = "Rp {:,.0f}"
        elif col.startswith("Δ_"):
            format_dict[col] = "{:+.1f}%"

    st.dataframe(full_table.style.format(format_dict))

else:
    st.warning("Kolom 'brand' atau 'total_sales' tidak ditemukan.")




# ====================
# 📊 Table Per Tahun
# ====================


with col1:
    st.subheader("Top Customers")
    if "customers" in df_filtered.columns:
        pivot = pd.pivot_table(df_filtered, index="customers", values=["qty", "total_sales"], aggfunc="sum")
        pivot = pivot.sort_values(by="total_sales", ascending=False)
        st.dataframe(pivot.style.format("{:,.0f}"))
    else:
        st.info("Kolom 'customers' tidak tersedia.")

with col2:
    st.subheader("Top Brand")
    if "brand" in df_filtered.columns:
        pivot = pd.pivot_table(df_filtered, index="brand", values=["qty", "total_sales"], aggfunc="sum")
        pivot = pivot.sort_values(by="total_sales", ascending=False)
        st.dataframe(pivot.style.format("{:,.0f}"))
    else:
        st.info("Kolom 'brand' tidak tersedia.")

with col3:
    st.subheader("Sales per Year")
    pivot = pd.pivot_table(df_filtered, index="year", values=["qty", "total_sales"], aggfunc="sum")
    st.dataframe(pivot.style.format("{:,.0f}"))

with col4:
    st.subheader("Sales per Month")
    pivot = pd.pivot_table(df_filtered, index="month", values=["qty", "total_sales"], aggfunc="sum")
    pivot = pivot.reindex(month_order)
    st.dataframe(pivot.style.format("{:,.0f}"))

# ====================
# VISUALISASI
# ====================
st.markdown("---")
st.subheader("Sales Visualizations")

chart_option = st.selectbox("Pilih data:", ["Total Sales", "Quantity", "Total Sales & Quantity"])

# Top Customers Chart
if "customers" in df_filtered.columns:
    st.markdown("**Top 10 Customers**")
    top_customers_df = df_filtered.groupby("customers")[["qty", "total_sales"]].sum().sort_values(by="total_sales", ascending=False).head(10).reset_index()

    if chart_option == "Total Sales":
        chart = alt.Chart(top_customers_df).mark_bar(color="#1f77b4").encode(
            x="total_sales:Q", y=alt.Y("customers:N", sort='-x'), tooltip=["customers", "total_sales"]
        )
    elif chart_option == "Quantity":
        chart = alt.Chart(top_customers_df).mark_bar(color="#ff7f0e").encode(
            x="qty:Q", y=alt.Y("customers:N", sort='-x'), tooltip=["customers", "qty"]
        )
    else:
        bar = alt.Chart(top_customers_df).mark_bar(color="#1f77b4").encode(
            x="total_sales:Q", y=alt.Y("customers:N", sort='-x'), tooltip=["customers", "total_sales"]
        )
        line = alt.Chart(top_customers_df).mark_line(color="#ff7f0e", point=True).encode(
            x="qty:Q", y=alt.Y("customers:N", sort='-x'), tooltip=["customers", "qty"]
        )
        chart = bar + line

    st.altair_chart(chart.properties(height=300), use_container_width=True)

# ====================
# Sales per Month Chart
# ====================

st.markdown("**Sales per Month**")
monthly_df = df_filtered.groupby("month")[["qty", "total_sales"]].sum().reset_index()
monthly_df["month_num"] = monthly_df["month"].apply(lambda x: month_order.index(x) + 1)
monthly_df = monthly_df.sort_values(by="month_num").drop("month_num", axis=1)

if chart_option == "Total Sales":
    chart = alt.Chart(monthly_df).mark_bar(color="#1f77b4").encode(
    x=alt.X("month:N", sort=month_order, axis=alt.Axis(labelAngle=-0)),
    y="total_sales:Q",
    tooltip=["month", "total_sales"]
    )
elif chart_option == "Quantity":
    chart = alt.Chart(monthly_df).mark_bar(color="#1f77b4").encode(
    x=alt.X("month:N", sort=month_order, axis=alt.Axis(labelAngle=-0)),
    y="total_sales:Q",
    tooltip=["month", "total_sales"]
    )
    chart = alt.Chart(monthly_df).mark_bar(color="#ff7f0e").encode(
        x="month:N", y="qty:Q", tooltip=["month", "qty"]
    )
else:
    bar = alt.Chart(monthly_df).mark_bar(color="#1f77b4").encode(
        x="month:N", y="total_sales:Q", tooltip=["month", "total_sales"]
    )
    line = alt.Chart(monthly_df).mark_line(color="#ff7f0e", point=True).encode(
        x="month:N", y="qty:Q", tooltip=["month", "qty"]
    )
    chart = bar + line

st.altair_chart(chart.properties(height=300), use_container_width=True)

# ====================
# 📈 Kenaikan Omzet per Tahun
# ====================
st.markdown("---")
st.subheader("Kenaikan Penjualan per Tahun")

if "total_sales" in df_filtered.columns and "year" in df_filtered.columns:
    sales_by_year = df_filtered.groupby("year")["total_sales"].sum().reset_index()

    # Format angka menjadi dalam juta / miliar jika perlu
    line_chart = alt.Chart(sales_by_year).mark_line(point=True, color="#2ca02c").encode(
        x=alt.X("year:O", title="Tahun",axis=alt.Axis(labelAngle=-0)),
        y=alt.Y("total_sales:Q", title="Total Sales", axis=alt.Axis(format=",.0f")),
        tooltip=[
            alt.Tooltip("year:O", title="Tahun"),
            alt.Tooltip("total_sales:Q", title="Total Sales", format=",.0f")
        ]
    ).properties(
        width=700,
        height=400,
        title="Trend Kenaikan Omzet per Tahun"
    )

    st.altair_chart(line_chart, use_container_width=True)
else:
    st.warning("Data tidak lengkap untuk membuat line chart omzet.")

st.markdown("**Penjualan per Department**")
if "dept" in df_filtered.columns:
    category_sales = df_filtered.groupby("dept")["total_sales"].sum().reset_index()
    chart = alt.Chart(category_sales).mark_bar().encode(
    x=alt.X("dept:N", axis=alt.Axis(labelAngle=0, title="Department")),
    y=alt.Y("total_sales:Q", title="Total Sales"),
    color="dept:N",
    tooltip=["dept", "total_sales"]
    ).properties(
    width=700,
    height=400,
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("Kolom 'dept' tidak tersedia.")


st.markdown("**Penjualan per Customers**")
if "dept" in df_filtered.columns:
    category_sales = df_filtered.groupby("customers")["total_sales"].sum().reset_index()
    chart = alt.Chart(category_sales).mark_bar().encode(
    x=alt.X("customers:N", axis=alt.Axis(labelAngle=0, title="Customers")),
    y=alt.Y("total_sales:Q", title="Total Sales"),
    color="customers:N",
    tooltip=["customers", "total_sales"]
    ).properties(
    width=700,
    height=400,
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.warning("Kolom 'customers' tidak tersedia.")