import shutil
import streamlit as st
import pandas as pd
import os
import json
import time
from itertools import cycle
import re
from datetime import datetime
from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from typing import List, Dict
from langchain_experimental.text_splitter import SemanticChunker
import hashlib
from core.chatbot import chatbot
from typing import List, Dict

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",
)

# 页面标题
st.subheader("文献处理")
st.divider()

# 初始化目录和文件
os.makedirs("PDF", exist_ok=True)
os.makedirs("Markdown", exist_ok=True)  # 新增Markdown存储目录
JSONL_PATH = "Jsonfile/metadata.jsonl"
Config_Path = "Jsonfile/em_model_config.json"
# 初始化session_state（从JSONL加载历史数据）
if "file_data" not in st.session_state:
    if os.path.exists(JSONL_PATH):
        with open(JSONL_PATH, "r") as f:
            data = [json.loads(line) for line in f]
            # 确保每个条目都有QA_result字段
            for entry in data:
                if "QA_result" not in entry:
                    entry["QA_result"] = []
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
            "QA_result",
            "Chunks地址"
            ])

class SmartDocumentProcessor:
    def __init__(self,config_path: str = "Jsonfile/em_model_config.json"):
        with open(config_path, 'r') as f:
            em_config = json.load(f)
        self.embed_model = HuggingFaceEmbeddings(
            # model_name="/home/binbin/deeplearning/MinerU/bge-m3",
            model_name=em_config["model_paths"]["embedding_model"],
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 10}
        )

    def _detect_content_type(self, text: str) -> str:
        """增强型内容类型检测"""
        if re.search(r'```[\s\S]*?```|def |import |print\(', text):
            return "code"
        if re.search(r'\|.*\|.*\n[-:|]+', text):
            return "table"
        if re.search(r'\$.*?\$|\\[({]', text):
            return "formula"
        return "normal"

    def process_documents(self, file_path: str) -> List[Dict]:
        loader = TextLoader(file_path)
        documents = loader.load()

        # 语义分块
        chunker = SemanticChunker(
            embeddings=self.embed_model,
            breakpoint_threshold_amount=82,
            add_start_index=True
        )
        base_chunks = chunker.split_documents(documents)

        # 动态二次分块
        final_chunks = []
        for chunk in base_chunks:
            content_type = self._detect_content_type(chunk.page_content)
            
            if content_type == "code":
                splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
            elif content_type == "table":
                splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=96)
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
                
            final_chunks.extend(splitter.split_documents([chunk]))

        # 添加嵌入元数据
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}_{hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]}",
                "content_type": self._detect_content_type(chunk.page_content),
                "source_file": file_path
            })
        return final_chunks
    
    def save_full_chunks(self, chunks: List[Dict], file_name: str) -> str:
        """保存所有chunks到单个json文件"""
        chunks_dir = Path("Chunks")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成唯一文件名
        file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
        chunk_name = f"{file_hash}_{Path(file_name).stem}.json"
        chunk_path = chunks_dir / chunk_name
        
        # 写入内容
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump([dict(chunk) for chunk in chunks], f, indent=2,ensure_ascii=False)  # 关键点：不添加任何额外字符
                
        return str(chunk_path)     

def article_split(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()
        # print(txt)
    # 使用#进行分割
    slice_ = re.split("#", txt)
    # 合并较短部分
    slice_1 = []
    temp = []
    for i in slice_:
        # 根据换行符数量判断段落长度
        if len(re.findall("\n", i)) <= 2:
            temp.append(i)
        elif len(temp) > 0:
            temp.append(i)
            slice_1.append(''.join(temp))
            temp = []
        else:
            slice_1.append(i)
    # 提取摘要、介绍、结论
    abstract = None
    introduction = None
    conclusion = None
    # 查找摘要
    for section in slice_1:
        if re.search(r"\babstract\b", section, re.IGNORECASE):
            abstract = section
            break
    # 查找引言
    for section in slice_1:
        if re.search(r"\bintroduction\b", section, re.IGNORECASE) and section != abstract:
            introduction = section
            break
    # 查找结论（倒序查找）
    for section in reversed(slice_1):
        if re.search(r"\bconclusion\b", section, re.IGNORECASE):
            conclusion = section
            break
    # 如果未找到结论，尝试在参考文献前查找
    if not conclusion:
        for i in range(len(slice_1)):
            if re.search(r"\breferences\b", slice_1[-1-i], re.IGNORECASE):
                if len(slice_1) > i + 1:
                    conclusion = slice_1[-2-i]
                    break
    return (abstract, introduction, conclusion)

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

def query_generation(entities):
    base_prompt = """\
## Question Generation Requirements:
1. Generate 10 independent questions based on the following academic entities. Questions must focus on the entity's definition, principles, or applications.
2. Strictly avoid mentioning paper content, author research, or contextual information.
3. Prohibit generating questions of the following types:
   - "What is the role of this entity in the paper?"
   - "How did the authors apply this entity?"
   - "Where is this entity mentioned in the paper?"
4. Question types should be diversified, including but not limited to:
   - Concept explanation (What is...)
   - Technical comparison (Differences from...)
   - Application  (How to apply)
   - Historical development (Evolution process)
   - Mathematical principles (How it works/calculates)

## Format Requirements:
["Question 1","Question 2",...,"Question 10"]
## Entity List:
{entities}
Output ONLY the JSON-formatted list of questions that meet the requirements, without any explanations.
"""
    
    formatted_prompt = base_prompt.format(
        entities="\n".join([f"- {e}" for e in entities])
    )
    # logging.info("开始生成问题，实体数量：%d", len(entities))
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response, _ = chatbot(formatted_prompt)
            
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            questions = json.loads(response[start_idx:end_idx])
            
            if len(questions) != 10:
                raise ValueError("生成问题数量不足")
                
            cleaned_questions = [
                q.replace("In this paper", "").replace("In this study", "")
                .replace("According to the paper", "").replace("As mentioned in the research", "")
                .replace("the authors'", "").replace("the proposed", "")
                .strip(' ,.;')
                for q in questions
            ]
            
            return cleaned_questions
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Attempt {attempt+1} failed, retrying...")
            if attempt == max_retries -1:
                templates = cycle([
                    "What is the scientific definition of {entity}?",
                    "How does {entity} fundamentally operate?",
                    "What are the key components of {entity}?",
                    "What problem space does {entity} primarily address?",
                    "What are the limitations of {entity}?"
                ])
                selected_entities = entities[:10] if len(entities) >= 10 else (entities * (10 // len(entities) + 1))[:10]
                return [
                    next(templates).format(entity=entity)
                    for entity in selected_entities
                ]


colA, colB, colC, colD, colE, colF = st.columns(
    [0.3, 0.14, 0.14, 0.14, 0.14, 0.14], 
    vertical_alignment="center"
)

with colA:
    uploaded_files = st.file_uploader("选择PDF文件", accept_multiple_files=True, type=["pdf"])
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_size = uploaded_file.size / (1024*1024)
        size_display = f"{file_size:.2f} MB"
        
        # 防止重复上传
        if not ((st.session_state.file_data["标题"] == file_name) & 
                (st.session_state.file_data["大小"] == size_display)).any():
            
            # 保存PDF文件
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            saved_name = f"{timestamp}_{file_name}"
            file_path = os.path.join("PDF", saved_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 构建元数据
            metadata = {
                "标题": file_name,
                "上传时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "大小": size_display,
                "状态": "已上传",
                "存储路径": file_path,
                "md路径": "" , # 新增markdown路径字段
                "向量库路径":"",
                "实体数量":0,
                "实体":[],
                "Chunks地址":"",
                "QA_result":[]
            }
            with open(JSONL_PATH, "a") as f:
                f.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            
            # 更新session_state
            st.session_state.file_data = pd.concat(
                [st.session_state.file_data, pd.DataFrame([metadata])],
                ignore_index=True
            )

with colB:
    if st.button("预处理", use_container_width=True):
        selection = st.session_state.get("data", {}).get("selection", {})
        selected_rows = selection.get("rows", [])
        
        if not selected_rows:
            st.warning("请先选择要预处理的文件")
        else:
            progress = st.progress(0)
            status = st.empty()
            errors = []
            total = len(selected_rows)
            
            for idx, row_idx in enumerate(selected_rows, 1):
                file_name = st.session_state.file_data.at[row_idx, "标题"]
                status.info(f"正在处理 {idx}/{total}: {file_name}")
                progress.progress((idx-1)/total)  # 先显示准备状态
                
                try:
                    # 实际处理过程开始
                    file_path = st.session_state.file_data.at[row_idx, "存储路径"]
                    name_without_suff = Path(file_name).stem
                    local_md_dir = "Markdown"
                    os.makedirs(local_md_dir, exist_ok=True)
                    
                    reader = FileBasedDataReader("")
                    pdf_bytes = reader.read(file_path)
                    ds = PymuDocDataset(pdf_bytes)
                    parse_method = ds.classify()
                    infer_result = ds.apply(doc_analyze, ocr=(parse_method == SupportedPdfParseMethod.OCR))
                    
                    if parse_method == SupportedPdfParseMethod.OCR:
                        pipe_result = infer_result.pipe_ocr_mode(None)
                    else:
                        pipe_result = infer_result.pipe_txt_mode(None)
                    
                    md_content = pipe_result.get_markdown("")
                    output_md_path = os.path.join(local_md_dir, f"{name_without_suff}.md")
                    
                    with open(output_md_path, "w", encoding="utf-8") as f:
                        f.write(md_content)
                    
                    # 更新元数据
                    st.session_state.file_data.at[row_idx, "状态"] = "已转换"
                    st.session_state.file_data.at[row_idx, "md路径"] = output_md_path
                    with open(JSONL_PATH, "w") as f:
                        for _, row in st.session_state.file_data.iterrows():
                            json.dump(row.to_dict(), f, ensure_ascii=False)
                            f.write("\n")
                    
                    # 处理完成后更新状态和进度
                    status.success(f"✅ {idx}/{total}: {file_name} 转换成功")
                    progress.progress(idx/total)
                    time.sleep(1.5)
                    
                except Exception as e:
                    error_msg = f"❌ {file_name}: {str(e)}"
                    errors.append(error_msg)
                    status.error(error_msg)
                    st.session_state.file_data.at[row_idx, "状态"] = "转换失败"
                    progress.progress(idx/total)  # 即使失败也要推进进度
                
                # 强制界面更新（移除sleep保证实时性）
                status.empty()
                status.write(" ")
            
            # 最终处理
            progress.empty()
            if errors:
                status.error(f"处理完成，成功 {total-len(errors)} 个，失败 {len(errors)} 个")
                with st.expander("查看错误详情"):
                    st.error("\n\n".join(errors))
            else:
                status.success("✅ 所有文件处理成功！")
            
            st.rerun()

with colC:
    if st.button("提取实体", use_container_width=True):
        selection = st.session_state.get("data", {}).get("selection", {})
        selected_rows = selection.get("rows", [])
        if not selected_rows:
            st.warning("请先选择要抽取实体的文件")
        else:
            progress = st.progress(0)
            status = st.empty()
            errors = []
            total = len(selected_rows)
            
            for idx, row_idx in enumerate(selected_rows, 1):
                try:
                    file_name = st.session_state.file_data.at[row_idx, "标题"]
                    md_path = st.session_state.file_data.at[row_idx, "md路径"]
                    # print(md_path)
                    
                    # 检查markdown文件是否存在
                    if not os.path.exists(md_path):
                        st.warning(f"Markdown文件 {md_path} 不存在，请先预处理！")
                        time.sleep(1.5)
                        
                    status.info(f"处理 {idx}/{total}: {file_name}")
                    progress.progress((idx-1)/total)
                    # 执行内容切分
                    abstract, intro, conclusion = article_split(md_path)
                    # print(abstract,intro,conclusion)
                    
                    # 构建提示词
                    prompt = """
                    ## Generation Requirements:
                    1. Please extract 15 entities from the above text.
                    2. The format of the generation is as follows:
                    ["Entity 1","Entity 2",...]
                    3. Do not generate any text that is not related to the generation requirements.
                    4. The extracted entities only include academic entities such as concepts and academic terms, and do not include entities that are not related to academics, such as names, dates, and places.
                    Abstract：{abstract}
                    Instruction：{intro}
                    Conclusion：{conclusion}
                    
                    """.format(
                        abstract=abstract or "None",
                        intro=intro or "None",
                        conclusion=conclusion or "None"
                    )
                    # print(prompt)
                    # 调用大模型API
                    entities_str, token_usage = chatbot(prompt)
                    entities = json.loads(entities_str)
                    
                    # 更新元数据
                    st.session_state.file_data.at[row_idx, "实体"] = entities
                    st.session_state.file_data.at[row_idx, "状态"] = "已抽取实体"
                    st.session_state.file_data.at[row_idx, "实体数量"] = len(entities)
                    
                    # 重写JSONL文件
                    with open(JSONL_PATH, "w", encoding="utf-8") as f:
                        for _, row in st.session_state.file_data.iterrows():
                            json.dump(row.to_dict(), f, ensure_ascii=False)
                            f.write("\n")
                            
                    status.success(f"✅ {idx}/{total}: {file_name} 抽取成功（{len(entities)}个实体）")
                    progress.progress(idx/total)
                    
                except Exception as e:
                    error_msg = f"❌ {file_name}: {str(e)}"
                    errors.append(error_msg)
                    st.session_state.file_data.at[row_idx, "状态"] = "抽取失败"
                    st.session_state.file_data.at[row_idx, "实体"] = []
                    progress.progress(idx/total)
                    status.error(error_msg)
                    continue
                finally:
                    time.sleep(0.5)  # 避免界面更新过快
                    
            # 最终处理
            progress.empty()
            if errors:
                status.error(f"处理完成，成功 {total - len(errors)} 个，失败 {len(errors)} 个")
                with st.expander("错误详情"):
                    st.error("\n".join(errors))
            else:
                status.success("✅ 所有文件实体抽取完成！")
            st.rerun()
with colD:
    if st.button("生成问题", use_container_width=True):
        selection = st.session_state.get("data", {}).get("selection", {})
        selected_rows = selection.get("rows", [])
        if not selected_rows:
            st.warning("请先选择要生成QA的文件")
        else:
            progress = st.progress(0)
            status = st.empty()
            errors = []
            total = len(selected_rows)
            
            for idx, row_idx in enumerate(selected_rows, 1):
                try:
                    file_name = st.session_state.file_data.at[row_idx, "标题"]
                    entities = st.session_state.file_data.at[row_idx, "实体"]
                    
                    if not entities or len(entities) == 0:
                        raise ValueError("未找到实体，请先抽取实体")
                        time.sleep(2)
                    
                    status.info(f"处理 {idx}/{total}: {file_name}")
                    progress.progress((idx-1)/total)
                    
                    # 生成问题
                    questions = query_generation(entities)
                    
                    # 转换为QA结构
                    qa_list = [{"question": q, "answer": "", "reference": ""} for q in questions]
                    
                    # 更新元数据
                    st.session_state.file_data.at[row_idx, "QA_result"] = qa_list
                    st.session_state.file_data.at[row_idx, "状态"] = "已生问题"
                    
                    # 更新JSONL文件
                    with open(JSONL_PATH, "w", encoding="utf-8") as f:
                        for _, row in st.session_state.file_data.iterrows():
                            json.dump(row.to_dict(), f, ensure_ascii=False)
                            f.write("\n")
                            
                    status.success(f"✅ {idx}/{total}: {file_name} 生成成功（{len(questions)}个问题）")
                    progress.progress(idx/total)
                    
                except Exception as e:
                    error_msg = f"❌ {file_name}: {str(e)}"
                    errors.append(error_msg)
                    st.session_state.file_data.at[row_idx, "状态"] = "Q生成失败"
                    progress.progress(idx/total)
                    status.error(error_msg)
                finally:
                    time.sleep(0.5)
                    
            progress.empty()
            if errors:
                status.error(f"处理完成，成功 {total - len(errors)} 个，失败 {len(errors)} 个")
                with st.expander("错误详情"):
                    st.error("\n".join(errors))
            else:
                status.success("✅ 所有文件QA生成完成!请进入QA管理页面查看")
                time.sleep(2)
            st.rerun()
with colE:
    if st.button("文本嵌入", use_container_width=True):
        selection = st.session_state.get("data", {}).get("selection", {})
        selected_rows = selection.get("rows", [])
        
        if not selected_rows:
            st.warning("请先选择要嵌入的文件")
        else:
            progress = st.progress(0)
            status = st.empty()
            errors = []
            total = len(selected_rows)
            
            for idx, row_idx in enumerate(selected_rows, 1):
                try:
                    file_name = st.session_state.file_data.at[row_idx, "标题"]
                    md_path = st.session_state.file_data.at[row_idx, "md路径"]
                    
                    # 检查预处理状态
                    if st.session_state.file_data.at[row_idx, "md路径"] == "":
                        raise ValueError("请先完成预处理")
                    
                    status.info(f"处理中 {idx}/{total}: {file_name}")
                    progress.progress((idx-1)/total)
                    
                    # 生成唯一向量库路径
                    file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
                    vector_db_path = f"vector_db/{file_hash}_{Path(file_name).stem}"
                    
                    # 处理文档
                    processor = SmartDocumentProcessor()
                    chunks = processor.process_documents(md_path)

                    chunks_json_path = processor.save_full_chunks(chunks, file_name)
                    st.session_state.file_data.at[row_idx, "Chunks地址"] = chunks_json_path
                    
                    # 创建向量库
                    with open(Config_Path, 'r') as f:
                        em_config = json.load(f)
                    Chroma.from_documents(
                        documents=chunks,
                        embedding=HuggingFaceEmbeddings(model_name=em_config["model_paths"]["embedding_model"]),
                        persist_directory=vector_db_path
                    )
                    
                    # 更新元数据
                    st.session_state.file_data.at[row_idx, "状态"] = "已嵌入"
                    st.session_state.file_data.at[row_idx, "向量库路径"] = vector_db_path
                    
                    # 更新JSONL
                    with open(JSONL_PATH, "w", encoding="utf-8") as f:
                        for _, row in st.session_state.file_data.iterrows():
                            json.dump(row.to_dict(), f, ensure_ascii=False)
                            f.write("\n")
                            
                    status.success(f"✅ {idx}/{total}: {file_name} 嵌入成功")
                    progress.progress(idx/total)
                    
                except Exception as e:
                    error_msg = f"❌ {file_name}: {str(e)}"
                    errors.append(error_msg)
                    st.session_state.file_data.at[row_idx, "状态"] = "嵌入失败"
                    progress.progress(idx/total)
                    status.error(error_msg)
                finally:
                    time.sleep(0.5)
                    
            progress.empty()
            if errors:
                status.error(f"处理完成，成功 {total-len(errors)} 个，失败 {len(errors)} 个")
                with st.expander("错误详情"):
                    st.error("\n".join(errors))
            else:
                    status.success("✅ 所有文件嵌入完成！")
            st.rerun()
with colF:
    if st.button("删除", use_container_width=True):
        selection = st.session_state.get("data", {}).get("selection", {})
        selected_rows = selection.get("rows", [])
        if not selected_rows:
            st.warning("请先选择要删除的文件")
        else:
            # 检查是否是最后一个数据
            is_last_record = len(st.session_state.file_data) == len(selected_rows)
            
            # 逆序删除避免索引错位
            for row_idx in sorted(selected_rows, reverse=True):
                # 删除PDF文件
                pdf_path = st.session_state.file_data.at[row_idx, "存储路径"]
                try:
                    os.remove(pdf_path)
                except Exception as e:
                    st.error(f"删除PDF文件 {pdf_path} 失败: {e}")
                
                # 删除Markdown文件
                md_path = st.session_state.file_data.at[row_idx, "md路径"]
                if md_path:  # 确保路径存在
                    try:
                        os.remove(md_path)
                    except FileNotFoundError:
                        st.warning(f"Markdown文件 {md_path} 不存在")
                    except Exception as e:
                        st.error(f"删除Markdown文件 {md_path} 失败: {e}")
                # 删除向量库
                vector_db_path = st.session_state.file_data.at[row_idx, "向量库路径"]
                if vector_db_path:  # 确保路径存在
                    try:
                        shutil.rmtree(vector_db_path)
                    except FileNotFoundError:
                        st.warning(f"向量库 {vector_db_path} 不存在")
                    except Exception as e:
                        st.error(f"删除向量库 {vector_db_path} 失败: {e}")

                chunks_txt_path = st.session_state.file_data.at[row_idx, "Chunks地址"]
                if chunks_txt_path: 
                    try:
                        os.remove(chunks_txt_path)
                    except FileNotFoundError:
                        st.warning(f"Chunks地址 {chunks_txt_path} 不存在")
                    except Exception as e:
                        st.error(f"删除Chunks地址 {chunks_txt_path} 失败: {e}")

                # 从session_state中移除记录
                st.session_state.file_data = st.session_state.file_data.drop(row_idx)
            
            # 重置索引并更新JSONL
            st.session_state.file_data = st.session_state.file_data.reset_index(drop=True)
            
            # 如果是最后一个记录，完全删除JSONL文件
            if is_last_record:
                try:
                    os.remove(JSONL_PATH)
                    st.success("所有文件已删除，元数据文件已清除")
                except Exception as e:
                    st.error(f"删除元数据文件失败: {e}")
            else:
                # 重写JSONL文件
                with open(JSONL_PATH, "w") as f:
                    for _, row in st.session_state.file_data.iterrows():
                        json.dump(row.to_dict(), f, ensure_ascii=False)
                        f.write("\n")
                st.success("选中的文件及关联文件已删除")
            
            st.rerun()  # 重新加载页面更新显示


display_columns = ["标题", "上传时间", "大小", "状态", "存储路径", "md路径","向量库路径","实体数量","实体"]
# 在显示表格前添加
st.session_state.file_data["实体"] = st.session_state.file_data["实体"].apply(lambda x: x if isinstance(x, list) else [])
pdftable = st.dataframe(
    st.session_state.file_data[display_columns],
    key="data",
    height=600,
    on_select="rerun",
    selection_mode=["multi-row", "multi-column"],
    hide_index=True
)

selection = st.session_state.get("data", {}).get("selection", {})
selected_rows = selection.get("rows", [])

# 修改后的预览部分代码
if selected_rows:
    # 强制单选处理（只取第一个选中的行）
    if len(selected_rows) > 1:
        st.warning("⚠️ 请选择单个文件进行预览")
        selected_rows = [selected_rows[0]]  # 强制取第一个选中项
        st.session_state.data["selection"]["rows"] = selected_rows  # 更新选中状态

    if st.button("预览选中文件"):
        st.session_state.show_preview = True

# 动态响应选择状态变化
if not selected_rows and st.session_state.get("show_preview"):
    st.session_state.show_preview = False
    st.rerun()

if st.session_state.get("show_preview"):
    # 获取选中文件信息
    selected_index = selected_rows[0]
    selected_file = st.session_state.file_data.iloc[selected_index].to_dict()
    
    # 预览面板
    with st.container(border=True):
        col_close = st.columns([0.95, 0.05])
        with col_close[1]:
            if st.button("✕", type="secondary"):
                st.session_state.show_preview = False
                st.rerun()
        
        # 双栏布局
        col_pdf, col_md = st.columns([0.5, 0.5], gap="small")
        
        with col_pdf:
            st.subheader("PDF")
            pdf_path = selected_file["存储路径"]
            
            if os.path.exists(pdf_path):
                try:
                    import fitz
                    
                    doc = fitz.open(pdf_path)
                    page_count = len(doc)
                    
                    # 更紧凑的页面控制
                    cols = st.columns([0.1, 0.3, 0.1, 0.5])
                    with cols[1]:
                        selected_page = st.number_input(
                            "页码",
                            min_value=1,
                            max_value=page_count,
                            value=1,
                            label_visibility="collapsed"
                        )
                    # 添加左右箭头按钮
                    with cols[0]:
                        if st.button("←"):
                            selected_page = max(1, selected_page-1)
                    
                    with cols[2]:
                        if st.button("→"):
                            selected_page = min(page_count, selected_page+1)
                    
                    # 显示当前页/总页数
                    cols[3].caption(f"Page {selected_page}/{page_count}")
                    
                    # 页面渲染
                    with st.spinner("正在渲染..."):
                        page = doc.load_page(selected_page-1)
                        pix = page.get_pixmap(dpi=150)  # 提高DPI
                        img_data = pix.tobytes()
                        st.image(img_data, use_container_width =True)

                except Exception as e:
                    st.error(f"PDF预览失败: {str(e)}")

        with col_md:
            st.subheader("Markdown")
            md_path = selected_file["md路径"]

            # if md_path and os.path.exists(md_path):
            #     # 初始化session状态
            #     if 'edited_md' not in st.session_state:
            #         with open(md_path, "r", encoding="utf-8") as f:
            #             st.session_state.edited_md = f.read()
            if md_path and os.path.exists(md_path):
                # 获取当前选中文件的MD路径
                current_md_path = selected_file["md路径"]
                
                # 当切换文件或首次加载时更新内容
                if 'preview_md_path' not in st.session_state or st.session_state.preview_md_path != current_md_path:
                    with open(current_md_path, "r", encoding="utf-8") as f:
                        st.session_state.edited_md = f.read()
                    st.session_state.preview_md_path = current_md_path  # 记录当前路径
                
                # 创建编辑界面
                with st.container(height=1025):

                    edited = st.text_area(
                        "编辑内容",
                        value=st.session_state.edited_md,
                        height=900,
                        label_visibility="collapsed",
                        key="md_editor"
                    )
                    
                    if st.button("💾 保存修改"):
                        try:
                            with open(md_path, "w", encoding="utf-8") as f:
                                f.write(edited)
                            st.session_state.edited_md = edited
                            st.success("保存成功！")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"保存失败: {str(e)}")
            else:
                st.warning("Markdown文件未生成")
