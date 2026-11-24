import streamlit as st

from modules.ui import initialize_session_state, render_dashboard, render_sidebar


st.set_page_config(
    page_title="Smart Vision AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()
render_sidebar()
render_dashboard()
