import streamlit as st

pages = {
    "Market": [
        st.Page("pages/market.py", title="Dashboard", icon="📊"),
        st.Page("pages/heatmap.py", title="Heat Map", icon="🔥"),
    ],
    "Forecasting": [
        st.Page("pages/forecast.py", title="Multiple Linear Regression", icon="📈"),
    ],
    # "Settings": [
    #     st.Page("pages/dataset.py", title="Dataset", icon="⚙️"),
    # ],
}

pg = st.navigation(pages)
pg.run()