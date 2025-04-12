import streamlit as st

pages = {
    "Dataset": [
        st.Page("pages/dataset.py", title="Upload Dataset"),
        st.Page("pages/preview.py",title="Preview Dataset"),
    ],
    "Market": [
        st.Page("pages/dashboard.py", title="Dashboard"),
        st.Page("pages/heatmap.py", title="Heat Map"),
    ],
    "Data ETL": [
        st.Page("pages/cleaning.py", title="Data Cleaning"),
    ],
    "Forecasting": [
        st.Page("pages/forecast.py", title="Coming Soon"),
    ],
    # "Settings": [
    #     st.Page("pages/dataset.py", title="Dataset", icon="⚙️"),
    # ],
}

pg = st.navigation(pages)
pg.run()