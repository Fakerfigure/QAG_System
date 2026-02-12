import streamlit as st

# IMPORTANT: set_page_config must be the FIRST Streamlit command
st.set_page_config(
    page_title="QAG_System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core.i18n import init_language, get_text, language_selector

# Initialize language
init_language()

LOGO = "images/QAG_system_logo.png"

# Dynamic page titles based on language
Preprocessing = st.Page("pages/Preprocessing.py", title=get_text("nav_preprocessing"))
QA_management = st.Page("pages/QA_management.py", title=get_text("nav_qa_management"))
DB_management = st.Page("pages/DB_management.py", title=get_text("nav_db_management"))
Model_management = st.Page("pages/Model_management.py", title=get_text("nav_model_management"))

with st.sidebar:
    st.logo(
        LOGO,
        size="large"
    )
    
    # Language selector
    st.divider()
    language_selector()
    st.divider()

    pg = st.navigation(
        [
            Preprocessing, 
            QA_management,
            DB_management,
            Model_management,
        ]
    )
pg.run()