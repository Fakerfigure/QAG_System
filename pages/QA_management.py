import streamlit as st
import pandas as pd
import os
import json
import time
from pathlib import Path
from typing import List, Dict
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain.docstore.document import Document  
import torch
import re
from core.chatbot import chatbot

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# 页面标题
st.subheader("QA管理")
st.divider()

# 初始化目录和文件
JSONL_PATH = "Jsonfile/metadata.jsonl"

# 初始化session_state（从JSONL加载历史数据）
if "file_data" not in st.session_state:
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, "r") as f:
            data = [json.loads(line) for line in f]
            # 数据兼容性处理
            for entry in data:
                entry.setdefault("QA_result", [])
            st.session_state.file_data = pd.DataFrame(data)
    else:
        st.session_state.file_data = pd.DataFrame(columns=[
            "标题", 
            "上传时间", 
            "大小", 
            "状态", 
            "存储路径",
            "md路径",
            "向量库路径",
            "实体数量", 
            "实体",
            "QA_result"
            "Chunks地址"
            ])

def save_to_jsonl():
    """将当前数据保存回JSONL文件"""
    with open(JSONL_PATH, "w") as f:
        for _, row in st.session_state.file_data.iterrows():
            json_line = json.dumps(row.to_dict(), ensure_ascii=False)
            f.write(json_line + "\n")

# 检索器类
class HybridRetriever:
    def __init__(self, chunks: List[Dict], persist_dir: str,config_path: str = "Jsonfile/em_model_config.json"):
        """初始化混合检索器"""
        # self.logger = logging.getLogger(self.__class__.__name__)
        with open(config_path, 'r') as f:
            em_config = json.load(f)
        
        # 初始化向量数据库
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            # embedding=HuggingFaceEmbeddings(model_name="/home/binbin/deeplearning/MinerU/bge-m3"),
            embedding=HuggingFaceEmbeddings(model_name=em_config["model_paths"]["embedding_model"]),
            persist_directory=persist_dir
        )
        
        # BM25检索器
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
        
        # 混合检索器
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]  
        )
        
        # 重排序模型
        self.reranker = CrossEncoder(
            # "/home/binbin/deeplearning/MinerU/bge-reranker-large", 
            em_config["model_paths"]["reranker_model"],
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """两阶段检索流程"""
        # self.logger.info(f"处理查询: {query}")
        
        # 第一阶段：混合检索
        docs = self.ensemble_retriever.get_relevant_documents(query)
        
        # 第二阶段：重排序
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        return [doc for doc, _ in ranked_docs[:top_k]]

# def chatbot(content):
#     """调用大模型API生成回答"""
#     try:
#         client = OpenAI(
#             api_key="sk-c577fd2f34ed4a828b5d6ce320c8fdde",
#             base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
#             http_client=httpx.Client(proxies="http://127.0.0.1:7897")
#         )

#         completion = client.chat.completions.create(
#             model="qwen-plus",
#             messages=[{'role': 'user', 'content': content}],
#             temperature=0.3
#         )
        
#         # elapsed_time = time.time() - start_time
#         answer = completion.choices[0].message.content
#         used_tokens = completion.usage.total_tokens
#         return answer, used_tokens
#     except Exception as e:
#         return f"请求失败: {str(e)}", 0

# RAG系统
class EnhancedRAG:
    def __init__(self, chunks: List[Dict], persist_dir: str):
        self.retriever = HybridRetriever(chunks, persist_dir)

    def generate_prompt(self, question: str, contexts: List[Dict]) -> str:
        """生成结构化提示"""
        context_entries = []
        for doc in contexts:
            meta = getattr(doc, 'metadata', {})
            context_entries.append(
                f"[来源：{meta.get('source_file', 'unknown')}，"
                f"类型：{meta.get('content_type', 'normal')}]\n"
                f"{doc.page_content}"
            )
        context_str = "\n\n".join(context_entries)

        return f"""# Role Definition
You are the Chief Expert of a world-leading economic management think tank, possessing the following core competencies:
1. Profound expertise in cutting-edge theories, empirical research methods, and policy analysis in economic management
2. Exceptional ability to translate complex economic models into actionable business insights
3. Precision in identifying statistical significance and practical relevance in research data
4. Cross-disciplinary integration capabilities (finance, econometrics, strategic management, etc.)

# Knowledge Repository
The following knowledge units have undergone rigorous academic validation:
{context_str}

# Problem Statement
"{question}"

# Generate evidence-based response using knowledge repository
"""

    def ask(self, question: str) -> dict:
        """问答流程"""
        contexts = self.retriever.retrieve(question)
        prompt = self.generate_prompt(question, contexts)
        
        answer, used_tokens = chatbot(prompt)
        
        # 构建引用来源
        references = []
        for doc in contexts:
            meta = getattr(doc, 'metadata', {})
            references.append(
                f"来源文件: {meta.get('source_file', 'unknown')}\n"
                f"内容类型: {meta.get('content_type', 'normal')}\n"
                f"原文片段: {doc.page_content}"
            )
        reference_str = "\n\n".join(references)
        
        return {
            "answer": answer.strip(),
            "reference": reference_str,
            # "tokens": used_tokens
        }

# 页面操作按钮
colA, colB, colC = st.columns([0.3, 0.3, 0.3], vertical_alignment="center")
with colA:
    generate_btn = st.button("生成答案", help="为选中问题生成答案")
    if generate_btn:
        processed_files = total_processed = 0
        
        with st.spinner("处理中..."):
            try:
                for idx, row in st.session_state.file_data.iterrows():
                    df_key = f"qa_df_{idx}"
                    selected_rows = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])
                    if not selected_rows:
                        continue

                    # 加载并转换chunks数据
                    chunks_path = row.get("Chunks地址", "")
                    if not os.path.exists(chunks_path):
                        st.warning(f"跳过 {row['标题']}: chunks文件不存在")
                        continue
                    
                    with open(chunks_path, "r") as f:
                        raw_chunks = json.load(f)
                        chunks = [
                            Document(
                                page_content=item["page_content"],
                                metadata=item.get("metadata", {})
                            ) for item in raw_chunks
                        ]

                    # 初始化RAG
                    rag = EnhancedRAG(chunks, row["向量库路径"])
                    
                    # 处理选中问题
                    updated_qa = row["QA_result"].copy()
                    for qa_idx in selected_rows:
                        if qa_idx < len(updated_qa):
                            result = rag.ask(updated_qa[qa_idx]["question"])
                            updated_qa[qa_idx].update(result)
                            total_processed += 1
                    
                    st.session_state.file_data.at[idx, "QA_result"] = updated_qa
                    processed_files += 1
                    
                save_to_jsonl()
                st.success(f"成功处理{processed_files}个文件，生成{total_processed}个答案")
                st.rerun()
                
            except Exception as e:
                st.error(f"处理失败: {str(e)}")
with colB:
    delete_btn = st.button("删除QA", type="secondary", help="删除选中文章的所有QA记录")
    # 删除逻辑处理
    if delete_btn:
        try:
            updated_data = []
            deleted_count = 0
            
            for idx, row in st.session_state.file_data.iterrows():
                df_key = f"qa_df_{idx}"
                
                # 获取当前文章的选中行
                if df_key in st.session_state:
                    selected_rows = st.session_state[df_key].get("selection", {}).get("rows", [])
                    
                    # 过滤保留未选中的QA项
                    if selected_rows:
                        new_qa = [qa for qa_idx, qa in enumerate(row["QA_result"]) if qa_idx not in selected_rows]
                        deleted_count += len(selected_rows)
                    else:
                        new_qa = row["QA_result"]
                    
                    # 更新行数据
                    updated_row = row.copy()
                    updated_row["QA_result"] = new_qa
                    updated_data.append(updated_row)
                else:
                    updated_data.append(row)
            
            if deleted_count > 0:
                # 更新session_state
                st.session_state.file_data = pd.DataFrame(updated_data)
                
                # 持久化保存
                save_to_jsonl()

                st.success(f"成功删除 {deleted_count} 个QA项")
                time.sleep(1.5)
                st.rerun()
            else:
                st.warning("未选择要删除的QA项")
                time.sleep(1.5)
                
        except Exception as e:
            st.error(f"删除操作失败: {str(e)}")
            time.sleep(1.5)
            import traceback
            traceback.print_exc()
with colC: 
    DBcreate = st.button("创建数据集", type="secondary", help="将选中的QA保存为新的数据集文件")
    if DBcreate:
        st.session_state.show_dataset_dialog = True  # 触发显示对话框

    # 处理数据集创建对话框
    if st.session_state.get("show_dataset_dialog"):
        with st.form(key="dataset_creation_form"):
            dataset_name = st.text_input("数据集名称（无需后缀）", key="dataset_name", 
                                       help="请输入英文名称，不要包含特殊字符")
                    # 使用列来横向排列按钮
            col1, col2 = st.columns([1, 3])  # 调整比例为1:3使按钮靠左
            with col1:
                submit = st.form_submit_button("创建")
            with col2:
                cancel = st.form_submit_button("取消")
            # submit = st.form_submit_button("创建")
            # cancel = st.form_submit_button("取消")
            if submit:
                if not dataset_name.strip():
                    st.error("数据集名称不能为空!")
                else:
                    # 收集所有选中的QA项
                    selected_entries = []
                    for idx, row in st.session_state.file_data.iterrows():
                        df_key = f"qa_df_{idx}"
                        if df_key in st.session_state:
                            selected_rows = st.session_state[df_key].get("selection", {}).get("rows", [])
                            for qa_idx in selected_rows:
                                if qa_idx < len(row["QA_result"]):
                                    qa = row["QA_result"][qa_idx]
                                    selected_entries.append({
                                        "标题": row["标题"],
                                        "question": qa["question"],
                                        "answer": qa["answer"]
                                    })
                    
                    if not selected_entries:
                        st.error("请至少选择一个QA项!")
                    else:
                        # 创建保存目录
                        save_dir = Path("DataBase_manage")
                        save_dir.mkdir(parents=True, exist_ok=True)
                        
                        # 生成合法文件名
                        valid_name = re.sub(r'[\\/:*?"<>|]', "", dataset_name.strip())
                        file_path = save_dir / f"{valid_name}.jsonl"
                        
                        # 写入JSONL文件
                        try:
                            with open(file_path, "w", encoding="utf-8") as f:
                                for entry in selected_entries:
                                    json_line = json.dumps(entry, ensure_ascii=False)
                                    f.write(json_line + "\n")
                            st.success(f"数据集已创建: {file_path}")
                            time.sleep(1.5)
                            del st.session_state.show_dataset_dialog
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败: {str(e)}")

            if cancel:
                del st.session_state.show_dataset_dialog
                st.rerun()


# 展示所有文章的QA数据
for idx, row in st.session_state.file_data.iterrows():
    with st.expander(f"📄 {row['标题']} - QA数量: {len(row['QA_result'])}", expanded=True):
        if not row.get("QA_result"):
            st.info("该文章尚未生成QA内容")
            continue
            
        # 转换为交互式DataFrame
        qa_df = pd.DataFrame(row["QA_result"])[["question", "answer", "reference"]]
        qa_df.columns = ["问题", "答案", "参考文献"]
        
        # 创建唯一键值
        df_key = f"qa_df_{idx}"
        
        # 显示可交互表格
        selection = st.dataframe(
            qa_df,
            use_container_width=True,
            key=df_key,
            on_select="rerun",
            selection_mode="multi-row",
            hide_index=True,
            column_config={
                "问题": {"width": "40%"},
                "答案": {"width": "40%"},
                "参考文献": {"width": "20%"}
            }
        )
        # 添加编辑功能
        df_key = f"qa_df_{idx}"
        selected_rows = st.session_state.get(df_key, {}).get("selection", {}).get("rows", [])

        # 判断当前是否处于编辑模式
        is_editing = st.session_state.get("editing", {}).get("article_idx") == idx

        if not is_editing:
            # 显示编辑按钮（仅当选中一个QA时）
            if len(selected_rows) == 1:
                edit_col, _ = st.columns([0.2, 0.8])
                with edit_col:
                    if st.button("双击编辑选中QA", key=f"edit_btn_{idx}"):
                        st.session_state.editing = {
                            "article_idx": idx,
                            "qa_idx": selected_rows[0],
                            "original_question": row["QA_result"][selected_rows[0]]["question"],
                            "original_answer": row["QA_result"][selected_rows[0]]["answer"]
                        }

        # 显示编辑表单
        if is_editing and st.session_state.editing["article_idx"] == idx:
            qa_idx = st.session_state.editing["qa_idx"]
            current_qa = row["QA_result"][qa_idx]

            with st.form(key=f"edit_form_{idx}"):
                # 创建两列布局

                new_question = st.text_area(
                        "问题编辑",
                        value=st.session_state.editing["original_question"],
                        height=90,
                        key=f"question_edit_{idx}"
                    )
                col1, col2 = st.columns([0.6, 0.4])

                with col1:
                    new_answer = st.text_area(
                        "答案编辑", 
                        value=st.session_state.editing["original_answer"],
                        height=300,
                        key=f"answer_edit_{idx}"
                    )

                with col2:
                    st.text_area(
                        "参考文献",
                        value=current_qa["reference"],
                        height=300,
                        disabled=True,
                        key=f"reference_edit_{idx}"
                    )
                    # st.markdown(f"```\n{current_qa['reference']}\n```", help="此内容不可编辑")

                # 按钮布局
                btn_col1, btn_col2,_,_,_ = st.columns([0.2, 0.2,0.2,0.2,0.2])
                with btn_col1:
                    save_btn = st.form_submit_button("💾 保存修改")
                with btn_col2:
                    cancel_btn = st.form_submit_button("✕")

                if save_btn:
                    # 执行数据更新
                    updated_qa = row["QA_result"].copy()
                    updated_qa[qa_idx]["question"] = new_question
                    updated_qa[qa_idx]["answer"] = new_answer
                    
                    # 更新session状态
                    st.session_state.file_data.at[idx, "QA_result"] = updated_qa
                    save_to_jsonl()  # 保存到文件
                    del st.session_state.editing  # 退出编辑模式
                    st.success("修改已保存！")
                    time.sleep(1)
                    st.rerun()

                if cancel_btn:
                    del st.session_state.editing
                    st.rerun()




