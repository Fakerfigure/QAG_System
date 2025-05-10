import streamlit as st

LOGO = "images/QAG_system_logo.png"

Preprocessing = st.Page("pages/Preprocessing.py",title="文献处理")
QA_management = st.Page("pages/QA_management.py",title="QA管理")
DB_management = st.Page("pages/DB_management.py",title="数据集管理")
Model_management = st.Page("pages/Model_management.py",title="模型管理")
# test_page = st.Page("pages/test.py",title="测试")

with st.sidebar:

    st.logo(
    LOGO,
    size="large"
    )

    pg = st.navigation(
        [
            Preprocessing, 
            QA_management,
            DB_management,
            Model_management,
        ]
    )
pg.run()