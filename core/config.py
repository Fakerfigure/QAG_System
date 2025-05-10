import streamlit as st
import json
from pathlib import Path

CONFIG_PATH = Path("Jsonfile/model_config.json")

def init_model_config():
    """初始化模型配置"""
    if "model_config" not in st.session_state:
        default_config = {
            "api_key": "",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-plus",
            "temperature": 0.3
        }
        try:
            if CONFIG_PATH.exists():
                with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                    saved_config = json.load(f)
                    # 合并配置保证新增字段兼容
                    merged_config = default_config.copy()
                    merged_config.update(saved_config)
                    st.session_state.model_config = merged_config
            else:
                st.session_state.model_config = default_config
        except Exception as e:
            st.error(f"配置加载失败: {str(e)}")
            st.session_state.model_config = default_config

def get_model_config():
    """获取当前配置"""
    init_model_config()  # 确保配置已初始化
    return st.session_state.model_config

def update_model_config(new_config):
    """更新配置（供配置页面使用）"""
    try:
        # 验证新配置是否为字典
        if not isinstance(new_config, dict):
            raise ValueError("新配置必须是一个字典对象")
        st.session_state.model_config = new_config
        # 异步保存到文件
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=4)
    except Exception as e:
        st.error(f"配置保存失败: {str(e)}")