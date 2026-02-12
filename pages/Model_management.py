from core.config import get_model_config, update_model_config
import streamlit as st
from core.chatbot import chatbot
from core.i18n import get_text, init_language

# Initialize language
init_language()

def config_page():
    st.subheader(get_text("model_title"))
    current_config = get_model_config()
    
    with st.form("config_form"):
        new_key = st.text_input(
            get_text("api_key"), 
            value=current_config["api_key"],
            type="password"
        )
        
        new_base_url = st.text_input(
            get_text("api_address"), 
            value=current_config["base_url"]
        )
        
        new_model = st.selectbox(
            get_text("model_selection"),
            options=["qwen-plus", "qwen-turbo", "qwen-max"],
            index=["qwen-plus", "qwen-turbo", "qwen-max"].index(current_config["model"])
        )
        
        new_temp = st.slider(
            get_text("temperature"), 
            min_value=0.0, 
            max_value=1.0, 
            value=current_config["temperature"]
        )
        
        colA, colB = st.columns([0.2, 0.2])
        
        with colA:
            if st.form_submit_button(get_text("save_config")):
                new_config = {
                    "api_key": new_key,
                    "base_url": new_base_url,
                    "model": new_model,
                    "temperature": new_temp
                }
                update_model_config(new_config)
                st.success(get_text("config_updated"))

        with colB:
            if st.form_submit_button(get_text("restore_default")):
                default_config = {
                    "api_key": "",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "model": "qwen-plus",
                    "temperature": 0.3
                }
                update_model_config(default_config)
                st.success(get_text("config_restored"))

def chat_page():
    config = get_model_config()
    
    st.subheader(get_text("llm_test"))
    st.write(f"{get_text('current_model')}：{config['model']} | {get_text('temperature')}：{config['temperature']}")
    
    user_input = st.text_input(get_text("input_question"))
    if user_input:
        answer, tokens = chatbot(user_input)
        st.text_area(get_text("response"), value=answer)
        st.write(f"{get_text('tokens_used')}：{tokens}")


config_page()
chat_page()