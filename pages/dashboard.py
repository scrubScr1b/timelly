import pandas as pd
import streamlit as st

st.title("Market Dashboard")

df = pd.read_excel("data/df_mz.xlsx")

with st.expander("Data Preview"):
    st.dataframe(df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("Top Customers")

    pivot = pd.pivot_table(
        df,
        index="Customers",
        values=["Qty", "Total_Sales"],
        aggfunc="sum",
    ).sort_values(by="Total_Sales", ascending=False)

    pivot = pivot.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})

    st.dataframe(pivot.style.format("{:,.0f}"))

with col2:
    st.subheader("Top Brand")

    pivot = pd.pivot_table(
        df,
        index="Brand",
        values=["Qty", "Total_Sales"],
        aggfunc="sum",
    ).sort_values(by="Total_Sales", ascending=False)

    pivot = pivot.rename(columns={"Qty": "Quantity", "Total_Sales": "Total Sales"})

    st.dataframe(pivot.style.format("{:,.0f}"))
