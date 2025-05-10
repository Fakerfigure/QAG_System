import streamlit as st
import pandas as pd
import json
import os
import time
from pathlib import Path

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# 页面标题
st.subheader("数据集管理")
st.divider()

# 初始化session状态
if "dataset_data" not in st.session_state:
    st.session_state.dataset_data = {}

if "show_export" not in st.session_state:
    st.session_state.show_export = False

# 数据集目录配置
DATASET_DIR = Path("DataBase_manage")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

def load_datasets():
    """加载或更新数据集数据"""
    dataset_files = list(DATASET_DIR.glob("*.jsonl"))
    
    # 移除已删除的数据集
    current_files = {f.name for f in dataset_files}
    for name in list(st.session_state.dataset_data.keys()):
        if name not in current_files:
            del st.session_state.dataset_data[name]
    
    # 加载/更新数据集
    for file in dataset_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            
            # 转换旧格式兼容
            for entry in data:
                if "标题" not in entry:
                    entry["标题"] = entry.get("source", "未知文档")
                
            df = pd.DataFrame(data)
            
            # 更新session状态
            st.session_state.dataset_data[file.name] = {
                "dataframe": df,
                "path": str(file.absolute()),
                "last_modified": file.stat().st_mtime
            }
        except Exception as e:
            st.error(f"加载数据集 {file.name} 失败: {str(e)}")

# 首次加载数据
load_datasets()

with st.sidebar:
    if st.button("📤 打开导出配置", help="配置数据集导出选项"):
        st.session_state.show_export = not st.session_state.show_export
    
    if st.session_state.show_export:
        with st.form("export_sidebar_config"):
            st.markdown("### 导出配置")
            
            # 文件格式选择
            file_format = st.selectbox(
                "文件格式", 
                ["JSON", "JSONL", "CSV"],
                index=0,
                key="export_format"
            )
            
            # 系统提示词输入
            system_prompt = st.text_input(
                "系统提示词",
                value="",
                key="system_prompt"
            )
            
            # # 高级选项
            # with st.expander("⚙️ 高级设置", expanded=False):
            #     st.checkbox("仅导出已确认数据", value=False, key="export_confirmed")
            #     st.checkbox("包含思维链", value=False, key="export_chain")
            
            # 格式示例
            st.markdown("​**​格式示例(Alpaca)​**​")
            example_data = {
                "instruction": "人类指令（必填）",
                "input": "人类输入（选填）",
                "output": "模型回答（必填）",
                "system": "系统提示词（选填）"
            }
            st.json(example_data)
            
            # 操作按钮
            col1, col2 = st.columns([1, 2])
            with col1:
                export_submit = st.form_submit_button("确认")
            with col2:
                if st.form_submit_button("取消"):
                    st.session_state.show_export = False
                    st.rerun()
            
            if export_submit:
                # 处理数据导出
                export_data = []
                for dataset_name, dataset_info in st.session_state.dataset_data.items():
                    df_key = f"dataset_table_{dataset_name}"
                    selected_indices = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])
                    
                    if not selected_indices:
                        continue
                    
                    dataset_df = dataset_info["dataframe"]
                    for idx in selected_indices:
                        row = dataset_df.iloc[idx]
                        export_entry = {
                            "instruction": row["question"],
                            "output": row["answer"],
                            "input": "",
                            "system": system_prompt if system_prompt else None
                        }
                        
                            
                        export_data.append(export_entry)
                
                if not export_data:
                    st.error("请至少选择一条要导出的数据！")
                else:
                    # 生成文件内容
                    try:
                        if file_format == "JSON":
                            content = json.dumps(export_data, ensure_ascii=False, indent=2)
                            mime_type = "application/json"
                        elif file_format == "JSONL":
                            content = "\n".join(json.dumps(item, ensure_ascii=False) for item in export_data)
                            mime_type = "application/json"
                        elif file_format == "CSV":
                            df = pd.DataFrame(export_data)
                            content = df.to_csv(index=False)
                            mime_type = "text/csv"
                        
                        # 存储导出信息
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        st.session_state.export_content = content
                        st.session_state.export_filename = f"dataset_{timestamp}.{file_format.lower()}"
                        st.session_state.export_mime = mime_type
                        
                    except Exception as e:
                        st.error(f"导出失败: {str(e)}")
def show_export_dialog():
    """显示导出配置对话框"""
    with st.form("export_config"):
        st.markdown("### 导出配置")
        
        # 文件格式选择
        file_format = st.selectbox(
            "文件格式", 
            ["JSON", "JSONL", "CSV"],
            key="export_format"
        )
        
        # 系统提示词输入
        system_prompt = st.text_input(
            "系统提示词",
            value="",
            key="system_prompt"
        )
        
        # 格式示例展示
        st.markdown("​**​格式示例​**​")
        example_data = {
            "instruction": "人类指令（必填）",
            "input": "",
            "output": "模型回答（必填）",
            "system": f"{system_prompt}(选填)"
        }
        st.json(example_data)
        
        # 操作按钮
        if st.form_submit_button("确认导出"):
            # 处理导出数据
            export_data = []
            for dataset_name, dataset_info in st.session_state.dataset_data.items():
                df_key = f"dataset_table_{dataset_name}"
                selected_indices = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])
                
                if not selected_indices:
                    continue
                
                dataset_df = dataset_info["dataframe"]
                for idx in selected_indices:
                    row = dataset_df.iloc[idx]
                    export_data.append({
                        "instruction": row["question"],
                        "input": "",
                        "output": row["answer"],
                        "system": system_prompt
                    })
            
            if not export_data:
                st.error("请至少选择一条要导出的数据！")
                return
            
            # 生成文件内容
            try:
                if file_format == "JSON":
                    content = json.dumps(export_data, ensure_ascii=False, indent=2)
                    mime_type = "application/json"
                elif file_format == "JSONL":
                    content = "\n".join(json.dumps(item, ensure_ascii=False) for item in export_data)
                    mime_type = "application/json"
                elif file_format == "CSV":
                    df = pd.DataFrame(export_data)
                    df = df[["instruction", "input", "output", "system"]]
                    content = df.to_csv(index=False)
                    mime_type = "text/csv"
                
                # 存储导出的数据到session
                st.session_state.export_content = content
                st.session_state.export_mime = mime_type
                st.session_state.export_filename = f"dataset_export_{time.strftime('%Y%m%d-%H%M%S')}.{file_format.lower()}"
                
            except Exception as e:
                st.error(f"导出失败: {str(e)}")
        
        if st.form_submit_button("取消"):
            st.session_state.show_export = False
            st.rerun()

def process_export(file_format, system_prompt):
    """处理导出逻辑"""
    export_data = []
    
    # 收集所有选中的数据
    for dataset_name, dataset_info in st.session_state.dataset_data.items():
        df_key = f"dataset_table_{dataset_name}"
        selected_indices = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])
        
        if not selected_indices:
            continue
        
        dataset_df = dataset_info["dataframe"]
        for idx in selected_indices:
            row = dataset_df.iloc[idx]
            export_data.append({
                "instruction": row["question"],
                "input": "",
                "output": row["answer"],
                "system": system_prompt
            })
    
    if not export_data:
        st.error("请至少选择一条要导出的数据！")
        return
    
    # 生成文件内容
    try:
        if file_format == "JSON":
            content = json.dumps(export_data, ensure_ascii=False, indent=2)
            mime_type = "application/json"
        elif file_format == "JSONL":
            content = "\n".join(json.dumps(item, ensure_ascii=False) for item in export_data)
            mime_type = "application/jsonl"
        elif file_format == "CSV":
            df = pd.DataFrame(export_data)
            df = df[["instruction", "input", "output", "system"]]  # 调整列顺序
            content = df.to_csv(index=False)
            mime_type = "text/csv"
        else:
            raise ValueError("不支持的导出格式")
        
        # 生成时间戳文件名
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"dataset_export_{timestamp}.{file_format.lower()}"
        
        # 提供下载
        st.download_button(
            label="下载导出文件",
            data=content,
            file_name=filename,
            mime=mime_type,
            key=f"download_{timestamp}"
        )
        
        # 关闭对话框
        st.session_state.show_export = False
        st.rerun()
        
    except Exception as e:
        st.error(f"导出失败: {str(e)}")

if st.session_state.dataset_data:
    # 在标题下方创建操作按钮区域
    header_col1, header_col2 = st.columns([0.2, 0.8])
    with header_col1:
        if st.button("删除所有选中项", help="注意：全选数据集将删除整个文件"):
            # 遍历所有数据集处理删除
            need_refresh = False
            
            for dataset_name, dataset_info in list(st.session_state.dataset_data.items()):
                df_key = f"dataset_table_{dataset_name}"
                selected_rows = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])
                
                if not selected_rows:
                    continue
                
                file_path = dataset_info["path"]
                df = dataset_info["dataframe"]
                total_qa = len(df)
                is_full_selection = len(selected_rows) == total_qa
                
                try:
                    if is_full_selection:
                        # 删除整个数据集
                        os.remove(file_path)
                        del st.session_state.dataset_data[dataset_name]
                        st.success(f"数据集 {dataset_name} 已永久删除")
                    else:
                        # 删除选中QA项
                        new_data = [
                            qa for idx, qa in 
                            enumerate(dataset_info["dataframe"].to_dict("records"))
                            if idx not in selected_rows
                        ]
                        
                        # 写入更新后的文件
                        with open(file_path, "w", encoding="utf-8") as f:
                            for entry in new_data:
                                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        
                        # 更新session状态
                        st.session_state.dataset_data[dataset_name]["dataframe"] = pd.DataFrame(new_data)
                        st.success(f"在 {dataset_name} 中删除 {len(selected_rows)} 个QA项")
                    
                    need_refresh = True
                    
                except Exception as e:
                    st.error(f"操作失败[{dataset_name}]: {str(e)}")
            
            if need_refresh:
                time.sleep(1)
                st.rerun()
            else:
                st.warning("没有选中任何需要删除的内容")

    with header_col2:
        # 显示下载按钮（当有导出内容时）
        if 'export_content' in st.session_state:
            st.download_button(
                label="⬇️ 下载数据集",
                data=st.session_state.export_content,
                file_name=st.session_state.export_filename,
                mime=st.session_state.export_mime,
                key="header_download",
                on_click=lambda: st.session_state.pop('export_content')
            )
        else:
            if st.button("📤 导出数据集", key="header_export_btn"):
                st.session_state.show_export = True

# 显示所有数据集
if not st.session_state.dataset_data:
    st.info("当前没有数据集，请使用QA管理页面创建新数据集")
else:
    for dataset_name, dataset_info in list(st.session_state.dataset_data.items()):
        df = dataset_info["dataframe"]
        file_path = dataset_info["path"]
        
        with st.expander(f"📚 {dataset_name} - 包含 {len(df)} 个QA对", expanded=True):
            # 显示元数据
            
            # 交互式表格
            df_key = f"dataset_table_{dataset_name}"
            selection = st.dataframe(
                df[["标题", "question", "answer"]],  # 显示关键字段
                column_config={
                    "标题": "来源文档",
                    "question": "问题",
                    "answer": "答案"
                },
                use_container_width=True,
                key=df_key,
                hide_index=True,
                height=400,
                on_select="rerun",
                selection_mode="multi-row"
            )