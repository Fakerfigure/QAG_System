<div align="center">

![QAG_System](images/QAG_System.png)

[English](README.md) | **中文**

</div>

# QAG_System

[QAG_System](https://github.com/Fakerfigure/QAG_System)是一个基于检索增强生成(RAG)技术的自动化问答数据集生成框架，专注于将科学论文转换成QA对，旨在为大型语言模型(LLM)训练提供高效、合规的高质量数据源。

## 目录

- [项目架构](#项目架构)
- [核心技术-Mini_RAG](#核心技术-mini_rag)
- [快速开始](#快速开始)
- [页面介绍](#页面介绍)
- [端到端流水线 (命令行)](#端到端流水线-命令行)
- [评估体系](#评估体系)

## 项目架构

![项目架构](images/qag_system.png)

整个系统的功能框架和文件结构如上图所示，我们将论文到QA的工作过程分为四个关键环节：

- **文献处理**：该环节主要完成文件的上传与解析，对文件进行预处理例如PDF转markdown、提取标题、摘要、介绍、结论等，此外该环节需要利用大模型提取实体并产生问题。
- **QA管理**：该环节主要完成基于Mini_RAG的答案生成，QA管理与编辑，以及完成初步的数据集创建。
- **数据集管理**：该环节主要完成数据集的管理与编辑，数据集的导出与下载。
- **模型管理**：该环节主要完成模型的API配置，模型参数调整以及模型测试。

## 核心技术-Mini_RAG

![Mini_RAG](images/miniRAG.png)

本项目为每个QAG任务创建了单独的RAG系统，可以在保证系统高效的同时便于管理不同的向量库，Mini_RAG具体的工作流程如下：

首先，从文档（.md 格式）中提取摘要、指令和结论等关键部分，并通过 LLM 抽取实体，生成实体列表。接着，利用 WH Prompt 和 LLM 生成与实体相关的问题集合。在检索阶段，结合向量数据库和混合检索器（mini_RAG），使用 BGE-M3 向量检索和 BM25 关键词检索的混合策略，对问题进行检索并排序，最终生成高质量的问答对。

## 快速开始

⚠️ **注意**：本项目依赖GPU加速，流畅运行本项目至少需要**8G**显存！！

### 拉取代码

```bash
git clone https://github.com/Fakerfigure/QAG_System.git
cd QAG_System
```

### Conda虚拟环境（可选）

```bash
conda create -n QAG_System python=3.10
conda activate QAG_System
```

### 安装 MinerU

本项目在预处理的PDF转Markdown环境整合MinerU，因此请移步[MinerU](https://github.com/opendatalab/MinerU)，先完成MinerU的安装，切记在同一个虚拟环境下。

### 安装依赖

```bash
pip install -r requirements.txt
```

### 模型下载

```bash
modelscope download --model BAAI/bge-m3 --local_dir ./dir
modelscope download --model BAAI/bge-reranker-large --local_dir ./dir
```

下载好模型后，需要手动将模型文件添加在 `Jsonfile/em_model_config.json` 中，假设路径为`bin/QAG_System/modelscope/BAAI/bge-m3`，则文件修改为：

```json
{
    "model_paths": {
        "embedding_model": "bin/QAG_System/modelscope/BAAI/bge-m3",
        "reranker_model": "bin/QAG_System/modelscope/BAAI/bge-reranker-large"
    }
}
```

### 启动

```bash
streamlit run QAG_System_app.py
```

启动成功后，若浏览器没有自动弹出，则手动访问 `http://localhost:8501`。

## 页面介绍

### 页面结构

```
QAG_System/
├── pages/
│   ├── Preprocessing.py
│   ├── QA_management.py
│   ├── DB_management.py
│   └── Model_management.py
└── QAG_System_app.py
```

基于streamlit的前端架构要求，所有页面都需要在`QAG_System_app.py`中进行注册，`QAG_System_app.py`中主要完成了系统的初始化和页面的调用以及导航栏的配置，其他页面都在`pages`目录下统一管理。

### 文献处理页

文献处理页主要包含文件的上传与解析，对文件进行预处理例如PDF转markdown、提取标题、摘要、介绍、结论等，此外该环节需要利用大模型提取实体并产生问题。

![文献处理](images/Preprocessing.png)

**文件上传**

用户可以点击或拖动上传论文文件（限定PDF格式），系统会为该文件创建元数据。

**预处理**

当用户选定一行或多行文件后，点击"预处理"按钮，系统会自动将选定的文件转换为markdown格式。

![预处理](images/yuchuli.png)

**Markdown预览与修改**

经过预处理的文档支持查看markdown，并修改markdown，选中需要查看的数据，在列表下方会出现"预览选中文件"，点击进入预览界面。

![markdown预览](images/markdown_edit.png)

**提取实体**

选择经过预处理的文件可以通过点击"提取实体"按钮来生成实体。

**生成问题**

选中完成抽取实体的文档可以进行问题生成，系统将会利用大模型生成10个围绕"实体"的科学问题。

**文本嵌入**

选择经过转换的文档，点击"文本嵌入"按钮，系统会将markdown文档进行切分并转换成向量存储在本地。

### QA管理页

该环节主要完成基于Mini_RAG的答案生成，QA管理与编辑，以及完成初步的数据集创建。

![QA管理页](images/qa_manage.png)

**生成答案**

选中需要生成答案的问题，点击"生成答案"按钮，系统会调取后台Mini_RAG框架来生成答案。

**QA编辑**

选中需要编辑的一组QA，用户可以自由调整除参考文献以外的问题和答案。

![QA编辑](images/qa_edit.png)

**创建数据集**

选择需要创建数据集的QA，点击"创建数据集"按钮，系统会弹出创建数据集的窗口。

![create_dataset](images/DBG.png)

### 数据集管理页

该页面主要完成数据集的管理与编辑，数据集的导出与下载。

![dataset_manage](images/DBM.png)

### 模型管理页

QAG_System每个页面或多或少都使用到了LLM，本系统提供一个模型管理页来对项目使用的LLM进行全局管理。

![model_manage_page](images/MM.png)

本系统仅支持类OpenAI式接口，用户可以根据自己的需求自行配置接口参数。

## 端到端流水线 (命令行)

如果您更喜欢命令行工具而非 WebUI，我们提供了一个端到端 QAG 流水线脚本，可以将单个 PDF 文件完整处理为 QA 数据集。

### 功能特点

- **完整流水线**：PDF → Markdown → 实体提取 → 问题生成 → 答案生成 (Mini_RAG)
- **性能指标**：记录每个阶段的耗时、Token 消耗和 GPU 显存占用
- **独立配置**：独立的配置文件，可单独设置 API 密钥和模型路径
- **Alpaca 格式输出**：生成可直接用于训练的 Alpaca 格式 QA 数据

### 使用方法

```bash
cd e2e_QAG

# 编辑 config.json 配置文件
# - api_key: 您的 LLM API 密钥
# - input_pdf: PDF 文件路径
# - embedding_model / reranker_model: 模型路径

python e2e_QAG.py
```

### 输出结果

执行完成后，查看 `output/` 目录：
- `markdown/` - 转换后的 Markdown 文件
- `vector_db/` - RAG 向量数据库
- `report.md` - 详细的性能报告，包含各阶段指标
- `report_alpaca.jsonl` - Alpaca 格式的 QA 数据

📊 **[查看示例报告](e2e_QAG/output/report.md)** - 包含从一篇研究论文生成的 10 个 QA 对的完整性能报告。

## 评估体系

系统包含5个评估维度：相关性、不可知性、完整性、准确性、合理性。

详见 [eval/readme.md](eval/readme.md) 获取详细的评估结果和提示词。
