"""
Internationalization (i18n) module for QAG_System
Provides multilingual support for the application
"""
import streamlit as st

# Language configuration
LANGUAGES = {
    "en": "English",
    "zh": "中文"
}

# Translation dictionary
TRANSLATIONS = {
    # Common
    "language": {
        "en": "Language",
        "zh": "语言"
    },
    
    # Navigation
    "nav_preprocessing": {
        "en": "Document Processing",
        "zh": "文献处理"
    },
    "nav_qa_management": {
        "en": "QA Management",
        "zh": "QA管理"
    },
    "nav_db_management": {
        "en": "Dataset Management",
        "zh": "数据集管理"
    },
    "nav_model_management": {
        "en": "Model Management",
        "zh": "模型管理"
    },
    
    # Document Processing Page
    "doc_title": {
        "en": "Document Processing",
        "zh": "文献处理"
    },
    "upload_file": {
        "en": "Choose PDF files",
        "zh": "选择PDF文件"
    },
    "preprocess": {
        "en": "Preprocess",
        "zh": "预处理"
    },
    "extract_entities": {
        "en": "Extract Entities",
        "zh": "提取实体"
    },
    "generate_questions": {
        "en": "Generate Questions",
        "zh": "生成问题"
    },
    "text_embedding": {
        "en": "Text Embedding",
        "zh": "文本嵌入"
    },
    "delete": {
        "en": "Delete",
        "zh": "删除"
    },
    "preview_file": {
        "en": "Preview Selected File",
        "zh": "预览选中文件"
    },
    "save_changes": {
        "en": "💾 Save Changes",
        "zh": "💾 保存修改"
    },
    
    # Table columns
    "col_title": {
        "en": "Title",
        "zh": "标题"
    },
    "col_upload_time": {
        "en": "Upload Time",
        "zh": "上传时间"
    },
    "col_size": {
        "en": "Size",
        "zh": "大小"
    },
    "col_status": {
        "en": "Status",
        "zh": "状态"
    },
    "col_storage_path": {
        "en": "Storage Path",
        "zh": "存储路径"
    },
    "col_md_path": {
        "en": "MD Path",
        "zh": "md路径"
    },
    "col_vector_path": {
        "en": "Vector DB Path",
        "zh": "向量库路径"
    },
    "col_entity_count": {
        "en": "Entity Count",
        "zh": "实体数量"
    },
    "col_entities": {
        "en": "Entities",
        "zh": "实体"
    },
    
    # Status
    "status_uploaded": {
        "en": "Uploaded",
        "zh": "已上传"
    },
    "status_converted": {
        "en": "Converted",
        "zh": "已转换"
    },
    "status_extracted": {
        "en": "Entities Extracted",
        "zh": "已抽取实体"
    },
    "status_questions_generated": {
        "en": "Questions Generated",
        "zh": "已生问题"
    },
    "status_embedded": {
        "en": "Embedded",
        "zh": "已嵌入"
    },
    
    # Messages
    "msg_select_files": {
        "en": "Please select files to preprocess first",
        "zh": "请先选择要预处理的文件"
    },
    "msg_select_extract": {
        "en": "Please select files to extract entities first",
        "zh": "请先选择要抽取实体的文件"
    },
    "msg_select_generate": {
        "en": "Please select files to generate QA first",
        "zh": "请先选择要生成QA的文件"
    },
    "msg_select_embed": {
        "en": "Please select files to embed first",
        "zh": "请先选择要嵌入的文件"
    },
    "msg_select_delete": {
        "en": "Please select files to delete first",
        "zh": "请先选择要删除的文件"
    },
    "msg_processing": {
        "en": "Processing",
        "zh": "正在处理"
    },
    "msg_processing_count": {
        "en": "Processing {0}/{1}: {2}",
        "zh": "正在处理 {0}/{1}: {2}"
    },
    "msg_success": {
        "en": "✅ All files processed successfully!",
        "zh": "✅ 所有文件处理成功！"
    },
    "msg_converted_success": {
        "en": "✅ {0}/{1}: {2} converted successfully",
        "zh": "✅ {0}/{1}: {2} 转换成功"
    },
    "msg_extracted_success": {
        "en": "✅ {0}/{1}: {2} extracted successfully ({3} entities)",
        "zh": "✅ {0}/{1}: {2} 抽取成功（{3}个实体）"
    },
    "msg_generated_success": {
        "en": "✅ {0}/{1}: {2} generated successfully ({3} questions)",
        "zh": "✅ {0}/{1}: {2} 生成成功（{3}个问题）"
    },
    "msg_embedded_success": {
        "en": "✅ {0}/{1}: {2} embedded successfully",
        "zh": "✅ {0}/{1}: {2} 嵌入成功"
    },
    "msg_select_single": {
        "en": "⚠️ Please select a single file to preview",
        "zh": "⚠️ 请选择单个文件进行预览"
    },
    "msg_no_preview": {
        "en": "Markdown file not generated",
        "zh": "Markdown文件未生成"
    },
    "msg_save_success": {
        "en": "Saved successfully!",
        "zh": "保存成功！"
    },
    "msg_deleted": {
        "en": "Selected files and related files have been deleted",
        "zh": "选中的文件及关联文件已删除"
    },
    "msg_all_deleted": {
        "en": "All files have been deleted, metadata file cleared",
        "zh": "所有文件已删除，元数据文件已清除"
    },
    
    # QA Management Page
    "qa_title": {
        "en": "QA Management",
        "zh": "QA管理"
    },
    "generate_answers": {
        "en": "Generate Answers",
        "zh": "生成答案"
    },
    "delete_qa": {
        "en": "Delete QA",
        "zh": "删除QA"
    },
    "create_dataset": {
        "en": "Create Dataset",
        "zh": "创建数据集"
    },
    "col_question": {
        "en": "Question",
        "zh": "问题"
    },
    "col_answer": {
        "en": "Answer",
        "zh": "答案"
    },
    "col_reference": {
        "en": "Reference",
        "zh": "参考文献"
    },
    "edit_qa": {
        "en": "Double-click to Edit Selected QA",
        "zh": "双击编辑选中QA"
    },
    "dataset_name": {
        "en": "Dataset Name (without extension)",
        "zh": "数据集名称（无需后缀）"
    },
    "create": {
        "en": "Create",
        "zh": "创建"
    },
    "cancel": {
        "en": "Cancel",
        "zh": "取消"
    },
    
    # Dataset Management Page
    "dataset_title": {
        "en": "Dataset Management",
        "zh": "数据集管理"
    },
    "export_config": {
        "en": "📤 Open Export Config",
        "zh": "📤 打开导出配置"
    },
    "export_format": {
        "en": "Export Format",
        "zh": "文件格式"
    },
    "file_format": {
        "en": "File Format",
        "zh": "文件格式"
    },
    "system_prompt": {
        "en": "System Prompt",
        "zh": "系统提示词"
    },
    "format_example": {
        "en": "**Format Example (Alpaca)**",
        "zh": "**格式示例(Alpaca)**"
    },
    "confirm": {
        "en": "Confirm",
        "zh": "确认"
    },
    "delete_selected": {
        "en": "Delete All Selected Items",
        "zh": "删除所有选中项"
    },
    "export_dataset": {
        "en": "📤 Export Dataset",
        "zh": "📤 导出数据集"
    },
    "download_dataset": {
        "en": "⬇️ Download Dataset",
        "zh": "⬇️ 下载数据集"
    },
    "source_document": {
        "en": "Source Document",
        "zh": "来源文档"
    },
    "no_dataset": {
        "en": "No datasets available. Please use QA Management page to create new datasets.",
        "zh": "当前没有数据集，请使用QA管理页面创建新数据集"
    },
    "export_config_title": {
        "en": "### Export Configuration",
        "zh": "### 导出配置"
    },
    "msg_select_export": {
        "en": "Please select at least one record to export!",
        "zh": "请至少选择一条要导出的数据！"
    },
    "msg_export_failed": {
        "en": "Export failed: {0}",
        "zh": "导出失败: {0}"
    },
    "msg_load_failed": {
        "en": "Failed to load dataset {0}: {1}",
        "zh": "加载数据集 {0} 失败: {1}"
    },
    "msg_dataset_deleted": {
        "en": "Dataset {0} has been permanently deleted",
        "zh": "数据集 {0} 已永久删除"
    },
    "msg_qa_deleted": {
        "en": "Deleted {0} QA items from {1}",
        "zh": "在 {0} 中删除 {1} 个QA项"
    },
    "msg_operation_failed": {
        "en": "Operation failed [{0}]: {1}",
        "zh": "操作失败[{0}]: {1}"
    },
    "msg_no_selection": {
        "en": "No items selected for deletion",
        "zh": "没有选中任何需要删除的内容"
    },
    "delete_selected_help": {
        "en": "Note: Selecting all items in a dataset will delete the entire file",
        "zh": "注意：全选数据集将删除整个文件"
    },
    "dataset_contains": {
        "en": "📚 {0} - Contains {1} QA pairs",
        "zh": "📚 {0} - 包含 {1} 个QA对"
    },
    "download_export": {
        "en": "Download Exported File",
        "zh": "下载导出文件"
    },
    "instruction_required": {
        "en": "Human instruction (required)",
        "zh": "人类指令（必填）"
    },
    "input_optional": {
        "en": "Human input (optional)",
        "zh": "人类输入（选填）"
    },
    "output_required": {
        "en": "Model response (required)",
        "zh": "模型回答（必填）"
    },
    "system_optional": {
        "en": "System prompt (optional)",
        "zh": "系统提示词（选填）"
    },
    "confirm_export": {
        "en": "Confirm Export",
        "zh": "确认导出"
    },
    "unknown_document": {
        "en": "Unknown Document",
        "zh": "未知文档"
    },
    
    # QA Management Page - additional
    "generate_answers": {
        "en": "Generate Answers",
        "zh": "生成答案"
    },
    "generate_answers_help": {
        "en": "Generate answers for selected questions",
        "zh": "为选中问题生成答案"
    },
    "delete_qa": {
        "en": "Delete QA",
        "zh": "删除QA"
    },
    "delete_qa_help": {
        "en": "Delete all QA records of selected articles",
        "zh": "删除选中文章的所有QA记录"
    },
    "create_dataset": {
        "en": "Create Dataset",
        "zh": "创建数据集"
    },
    "create_dataset_help": {
        "en": "Save selected QA as a new dataset file",
        "zh": "将选中的QA保存为新的数据集文件"
    },
    "dataset_name": {
        "en": "Dataset Name (without extension)",
        "zh": "数据集名称（无需后缀）"
    },
    "dataset_name_help": {
        "en": "Please enter an English name without special characters",
        "zh": "请输入英文名称，不要包含特殊字符"
    },
    "processing": {
        "en": "Processing...",
        "zh": "处理中..."
    },
    "msg_skip_file": {
        "en": "Skipping {0}: chunks file does not exist",
        "zh": "跳过 {0}: chunks文件不存在"
    },
    "msg_process_success": {
        "en": "Successfully processed {0} files, generated {1} answers",
        "zh": "成功处理{0}个文件，生成{1}个答案"
    },
    "msg_process_failed": {
        "en": "Processing failed: {0}",
        "zh": "处理失败: {0}"
    },
    "msg_delete_success": {
        "en": "Successfully deleted {0} QA items",
        "zh": "成功删除 {0} 个QA项"
    },
    "msg_no_qa_selected": {
        "en": "No QA items selected for deletion",
        "zh": "未选择要删除的QA项"
    },
    "msg_delete_failed": {
        "en": "Delete operation failed: {0}",
        "zh": "删除操作失败: {0}"
    },
    "msg_dataset_name_empty": {
        "en": "Dataset name cannot be empty!",
        "zh": "数据集名称不能为空!"
    },
    "msg_select_qa": {
        "en": "Please select at least one QA item!",
        "zh": "请至少选择一个QA项!"
    },
    "msg_dataset_created": {
        "en": "Dataset created: {0}",
        "zh": "数据集已创建: {0}"
    },
    "msg_save_failed": {
        "en": "Save failed: {0}",
        "zh": "保存失败: {0}"
    },
    "qa_count": {
        "en": "📄 {0} - QA Count: {1}",
        "zh": "📄 {0} - QA数量: {1}"
    },
    "no_qa_content": {
        "en": "This article has no QA content yet",
        "zh": "该文章尚未生成QA内容"
    },
    "edit_question": {
        "en": "Edit Question",
        "zh": "问题编辑"
    },
    "edit_answer": {
        "en": "Edit Answer",
        "zh": "答案编辑"
    },
    "msg_changes_saved": {
        "en": "Changes saved!",
        "zh": "修改已保存！"
    },
    
    # Preprocessing Page - additional
    "msg_no_entity": {
        "en": "No entities found, please extract entities first",
        "zh": "未找到实体，请先抽取实体"
    },
    "msg_need_preprocess": {
        "en": "Please complete preprocessing first",
        "zh": "请先完成预处理"
    },
    "msg_md_not_exist": {
        "en": "Markdown file {0} does not exist, please preprocess first!",
        "zh": "Markdown文件 {0} 不存在，请先预处理！"
    },
    "msg_complete_with_errors": {
        "en": "Completed, success: {0}, failed: {1}",
        "zh": "处理完成，成功 {0} 个，失败 {1} 个"
    },
    "view_error_details": {
        "en": "View Error Details",
        "zh": "查看错误详情"
    },
    "msg_all_convert_success": {
        "en": "✅ All files processed successfully!",
        "zh": "✅ 所有文件处理成功！"
    },
    "msg_all_extract_success": {
        "en": "✅ All files entity extraction completed!",
        "zh": "✅ 所有文件实体抽取完成！"
    },
    "msg_all_qa_success": {
        "en": "✅ All files QA generation completed! Please go to QA Management page",
        "zh": "✅ 所有文件QA生成完成!请进入QA管理页面查看"
    },
    "msg_all_embed_success": {
        "en": "✅ All files embedding completed!",
        "zh": "✅ 所有文件嵌入完成！"
    },
    "msg_delete_pdf_failed": {
        "en": "Failed to delete PDF file {0}: {1}",
        "zh": "删除PDF文件 {0} 失败: {1}"
    },
    "msg_delete_md_failed": {
        "en": "Failed to delete Markdown file {0}: {1}",
        "zh": "删除Markdown文件 {0} 失败: {1}"
    },
    "msg_delete_vector_failed": {
        "en": "Failed to delete vector database {0}: {1}",
        "zh": "删除向量库 {0} 失败: {1}"
    },
    "msg_delete_chunks_failed": {
        "en": "Failed to delete chunks file {0}: {1}",
        "zh": "删除Chunks地址 {0} 失败: {1}"
    },
    "msg_file_not_exist": {
        "en": "{0} does not exist",
        "zh": "{0} 不存在"
    },
    "msg_delete_metadata_failed": {
        "en": "Failed to delete metadata file: {0}",
        "zh": "删除元数据文件失败: {0}"
    },
    "rendering": {
        "en": "Rendering...",
        "zh": "正在渲染..."
    },
    "page_num": {
        "en": "Page",
        "zh": "页码"
    },
    "edit_content": {
        "en": "Edit Content",
        "zh": "编辑内容"
    },
    "pdf_preview_failed": {
        "en": "PDF preview failed: {0}",
        "zh": "PDF预览失败: {0}"
    },
    "md_not_generated": {
        "en": "Markdown file not generated",
        "zh": "Markdown文件未生成"
    },
    
    # Model Management Page
    "model_title": {
        "en": "LLM Configuration",
        "zh": "大模型配置"
    },
    "api_key": {
        "en": "API Key",
        "zh": "API Key"
    },
    "api_address": {
        "en": "API Address",
        "zh": "API地址"
    },
    "model_selection": {
        "en": "Model Selection",
        "zh": "模型选择"
    },
    "temperature": {
        "en": "Temperature",
        "zh": "温度参数"
    },
    "save_config": {
        "en": "Save Configuration",
        "zh": "保存配置"
    },
    "restore_default": {
        "en": "Restore Default",
        "zh": "恢复默认"
    },
    "config_updated": {
        "en": "Configuration updated",
        "zh": "配置已更新"
    },
    "config_restored": {
        "en": "Default configuration restored",
        "zh": "已恢复默认配置"
    },
    "llm_test": {
        "en": "LLM Test",
        "zh": "LLM测试"
    },
    "current_model": {
        "en": "Current Model",
        "zh": "当前模型"
    },
    "input_question": {
        "en": "Please enter your question:",
        "zh": "请输入问题："
    },
    "response": {
        "en": "Response:",
        "zh": "回答："
    },
    "tokens_used": {
        "en": "Tokens Used",
        "zh": "消耗token数"
    },
}


def init_language():
    """Initialize language setting in session state"""
    try:
        if "language" not in st.session_state:
            st.session_state.language = "zh"  # Default to Chinese
    except AttributeError:
        # Not running in streamlit context, skip initialization
        pass


def get_text(key: str, *args) -> str:
    """
    Get translated text for the current language with optional formatting
    
    Args:
        key: Translation key
        *args: Format arguments for string formatting
        
    Returns:
        Translated text, or key itself if not found
    """
    init_language()
    
    # Default to Chinese if not in streamlit context
    try:
        lang = st.session_state.language
    except (AttributeError, KeyError):
        lang = "zh"
    
    if key in TRANSLATIONS:
        text = TRANSLATIONS[key].get(lang, key)
        # Apply formatting if arguments provided
        if args:
            try:
                return text.format(*args)
            except (IndexError, KeyError):
                return text
        return text
    return key


def language_selector():
    """Display language selector in sidebar"""
    init_language()
    
    current_lang = st.session_state.language
    
    # Create language selector
    selected = st.selectbox(
        get_text("language"),
        options=list(LANGUAGES.keys()),
        format_func=lambda x: LANGUAGES[x],
        index=list(LANGUAGES.keys()).index(current_lang),
        key="lang_selector"
    )
    
    # Update language if changed
    if selected != st.session_state.language:
        st.session_state.language = selected
        st.rerun()
