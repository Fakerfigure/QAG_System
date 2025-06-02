from core.config import get_model_config, update_model_config
import streamlit as st
from core.chatbot import chatbot

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

def config_page():
    st.subheader("大模型配置")
    current_config = get_model_config()
    
    with st.form("config_form"):
        new_key = st.text_input("API Key", 
                               value=current_config["api_key"],
                               type="password")
        
        new_base_url = st.text_input("API地址", 
                                    value=current_config["base_url"])
        
        new_model = st.selectbox("模型选择",
                                options=["qwen-plus", "qwen-turbo", "qwen-max"],
                                index=["qwen-plus", "qwen-turbo", "qwen-max"].index(current_config["model"]))
        
        new_temp = st.slider("温度参数", 
                            min_value=0.0, 
                            max_value=1.0, 
                            value=current_config["temperature"])
        
        colA, colB = st.columns([0.2, 0.2])
        
        with colA:
            if st.form_submit_button("保存配置"):
                new_config = {
                    "api_key": new_key,
                    "base_url": new_base_url,
                    "model": new_model,
                    "temperature": new_temp
                }
                update_model_config(new_config)
                st.success("配置已更新")

        with colB:
            if st.form_submit_button("恢复默认"):
                default_config = {
                    "api_key": "",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen-plus",
                    "temperature": 0.3
                }
                update_model_config(default_config)
                st.success("已恢复默认配置")

def chat_page():
    config = get_model_config()
    
    st.subheader("LLM测试")
    st.write(f"当前模型：{config['model']} | temperature：{config['temperature']}")
    
    user_input = st.text_input("请输入问题：")
    if user_input:
        answer, tokens = chatbot(user_input)
        st.text_area("回答：", value=answer)
        st.write(f"消耗token数：{tokens}")


config_page()
chat_page()