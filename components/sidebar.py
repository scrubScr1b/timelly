# sidebar.py

import streamlit as st

def render_sidebar(pages):
    with st.sidebar:
        st.markdown("## 📂 Navigasi")
        kategori = st.selectbox("Pilih Kategori", list(pages.keys()))
        page_options = pages[kategori]

        if isinstance(page_options, list):
            judul = st.selectbox("Pilih Halaman", [p.title for p in page_options])
            for p in page_options:
                if p.title == judul:
                    p.render()
        else:
            page_options.render()
