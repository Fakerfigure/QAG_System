#!/usr/bin/env python3
"""
QAG_System End-to-End Benchmark
================================
End-to-end benchmark for the QAG pipeline:
PDF -> Markdown -> Entity Extraction -> Question Generation -> Answer Generation (Mini_RAG)

Usage:
    1. Edit config.json with your API key, input PDF path, and model paths
    2. Run: python benchmark.py
    3. Check output/report.md for results
"""

import os
import sys
import json
import time
import hashlib
import re
import psutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from itertools import cycle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import httpx
from openai import OpenAI
import gc

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# MinerU imports
from magic_pdf.data.data_reader_writer import FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

# LangChain imports
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_experimental.text_splitter import SemanticChunker
from langchain.docstore.document import Document
from sentence_transformers import CrossEncoder


# ============== Configuration ==============
@dataclass
class BenchmarkConfig:
    api_key: str = ""
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model: str = "qwen-plus"
    temperature: float = 0.3
    input_pdf: str = ""
    embedding_model: str = ""
    reranker_model: str = ""
    
    @classmethod
    def from_json(cls, path: str) -> "BenchmarkConfig":
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Check if model paths are valid, if not, fallback to em_model_config.json
        need_fallback = False
        if not data.get("embedding_model") or not os.path.exists(data.get("embedding_model", "")):
            need_fallback = True
        if not data.get("reranker_model") or not os.path.exists(data.get("reranker_model", "")):
            need_fallback = True
        
        if need_fallback:
            em_config_path = Path(__file__).parent.parent / "Jsonfile" / "em_model_config.json"
            if em_config_path.exists():
                print(f"⚠️  Model paths in config.json not found, loading from {em_config_path}")
                with open(em_config_path, 'r') as f:
                    em_config = json.load(f)
                data["embedding_model"] = em_config["model_paths"]["embedding_model"]
                data["reranker_model"] = em_config["model_paths"]["reranker_model"]
        
        return cls(**data)


# ============== Metrics Tracking ==============
@dataclass
class StageMetrics:
    name: str
    start_time: float = 0
    end_time: float = 0
    duration: float = 0
    tokens_used: int = 0
    gpu_memory_mb: float = 0
    cpu_percent: float = 0
    ram_used_mb: float = 0
    output_data: any = None
    error: str = ""
    
    def start(self):
        self.start_time = time.time()
        self._record_system_metrics()
    
    def stop(self, tokens: int = 0):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.tokens_used = tokens
        self._record_system_metrics()
    
    def _record_system_metrics(self):
        # CPU and RAM
        self.cpu_percent = psutil.cpu_percent()
        self.ram_used_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # GPU Memory
        if torch.cuda.is_available():
            self.gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)


@dataclass
class BenchmarkResult:
    total_start_time: float = 0
    total_end_time: float = 0
    total_duration: float = 0
    total_tokens: int = 0
    stages: List[StageMetrics] = field(default_factory=list)
    config: BenchmarkConfig = None
    pdf_name: str = ""
    entities: List[str] = field(default_factory=list)
    questions: List[str] = field(default_factory=list)
    qa_pairs: List[Dict] = field(default_factory=list)
    peak_gpu_memory_mb: float = 0
    
    def add_stage(self, stage: StageMetrics):
        self.stages.append(stage)
        self.total_tokens += stage.tokens_used
        if stage.gpu_memory_mb > self.peak_gpu_memory_mb:
            self.peak_gpu_memory_mb = stage.gpu_memory_mb


# ============== LLM Client ==============
class LLMClient:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=httpx.Client(timeout=60)
        )
    
    def chat(self, content: str) -> Tuple[str, int]:
        """Call LLM and return (response, tokens_used)"""
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[{'role': 'user', 'content': content}],
                temperature=self.config.temperature
            )
            answer = completion.choices[0].message.content
            tokens = completion.usage.total_tokens
            return answer, tokens
        except Exception as e:
            return f"Error: {str(e)}", 0


# ============== Document Processor ==============
class DocumentProcessor:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.embed_model = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"batch_size": 10}
        )
    
    def _detect_content_type(self, text: str) -> str:
        if re.search(r'```[\s\S]*?```|def |import |print\(', text):
            return "code"
        if re.search(r'\|.*\|.*\n[-:|]+', text):
            return "table"
        if re.search(r'\$.*?\$|\\[({]', text):
            return "formula"
        return "normal"
    
    def process_documents(self, file_path: str) -> List[Document]:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Semantic chunking
        chunker = SemanticChunker(
            embeddings=self.embed_model,
            breakpoint_threshold_amount=82,
            add_start_index=True
        )
        base_chunks = chunker.split_documents(documents)
        
        # Dynamic secondary chunking
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
        
        # Add metadata
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}_{hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]}",
                "content_type": self._detect_content_type(chunk.page_content),
                "source_file": file_path
            })
        
        return final_chunks


# ============== Mini RAG System ==============
class HybridRetriever:
    def __init__(self, chunks: List[Document], persist_dir: str, config: BenchmarkConfig, embed_model=None):
        # Reuse embedding model if provided
        if embed_model is None:
            self.embed_model = HuggingFaceEmbeddings(
                model_name=config.embedding_model,
                model_kwargs={"device": "cuda"}
            )
        else:
            self.embed_model = embed_model
        
        # Vector DB
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embed_model,
            persist_directory=persist_dir
        )
        
        # BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=5)
        
        # Ensemble retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )
        
        # Reranker
        self.reranker = CrossEncoder(
            config.reranker_model,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        docs = self.ensemble_retriever.get_relevant_documents(query)
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked_docs[:top_k]]


class MiniRAG:
    def __init__(self, chunks: List[Document], persist_dir: str, config: BenchmarkConfig, llm: LLMClient, embed_model=None):
        self.retriever = HybridRetriever(chunks, persist_dir, config, embed_model)
        self.llm = llm
    
    def generate_prompt(self, question: str, contexts: List[Document]) -> str:
        context_entries = []
        for doc in contexts:
            meta = getattr(doc, 'metadata', {})
            context_entries.append(
                f"[Source: {meta.get('source_file', 'unknown')}, "
                f"Type: {meta.get('content_type', 'normal')}]\n"
                f"{doc.page_content}"
            )
        context_str = "\n\n".join(context_entries)
        
        return f"""# Role Definition
You are the Chief Expert of a world-leading academic think tank, possessing the following core competencies:
1. Profound expertise in cutting-edge theories, empirical research methods, and policy analysis
2. Exceptional ability to translate complex models into actionable insights
3. Precision in identifying statistical significance and practical relevance in research data
4. Cross-disciplinary integration capabilities

# Knowledge Repository
The following knowledge units have undergone rigorous academic validation:
{context_str}

# Problem Statement
"{question}"

# Generate evidence-based response using knowledge repository
"""
    
    def ask(self, question: str) -> Tuple[Dict, int]:
        contexts = self.retriever.retrieve(question)
        prompt = self.generate_prompt(question, contexts)
        answer, tokens = self.llm.chat(prompt)
        
        references = []
        for doc in contexts:
            meta = getattr(doc, 'metadata', {})
            references.append(
                f"Source: {meta.get('source_file', 'unknown')}\n"
                f"Type: {meta.get('content_type', 'normal')}\n"
                f"Content: {doc.page_content[:200]}..."
            )
        
        return {
            "question": question,
            "answer": answer.strip(),
            "reference": "\n\n".join(references)
        }, tokens


# ============== Pipeline Functions ==============
def convert_pdf_to_markdown(pdf_path: str, output_dir: str) -> str:
    """Stage 1: PDF to Markdown conversion using MinerU"""
    reader = FileBasedDataReader("")
    pdf_bytes = reader.read(pdf_path)
    ds = PymuDocDataset(pdf_bytes)
    parse_method = ds.classify()
    infer_result = ds.apply(doc_analyze, ocr=(parse_method == SupportedPdfParseMethod.OCR))
    
    if parse_method == SupportedPdfParseMethod.OCR:
        pipe_result = infer_result.pipe_ocr_mode(None)
    else:
        pipe_result = infer_result.pipe_txt_mode(None)
    
    md_content = pipe_result.get_markdown("")
    
    # Save markdown
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, Path(pdf_path).stem + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    return md_path


def extract_sections(md_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Stage 2: Extract abstract, introduction, conclusion"""
    with open(md_path, 'r', encoding='utf-8') as f:
        txt = f.read()
    
    slice_ = re.split("#", txt)
    slice_1 = []
    temp = []
    
    for i in slice_:
        if len(re.findall("\n", i)) <= 2:
            temp.append(i)
        elif len(temp) > 0:
            temp.append(i)
            slice_1.append(''.join(temp))
            temp = []
        else:
            slice_1.append(i)
    
    abstract = introduction = conclusion = None
    
    for section in slice_1:
        if re.search(r"\babstract\b", section, re.IGNORECASE):
            abstract = section
            break
    
    for section in slice_1:
        if re.search(r"\bintroduction\b", section, re.IGNORECASE) and section != abstract:
            introduction = section
            break
    
    for section in reversed(slice_1):
        if re.search(r"\bconclusion\b", section, re.IGNORECASE):
            conclusion = section
            break
    
    if not conclusion:
        for i in range(len(slice_1)):
            if re.search(r"\breferences\b", slice_1[-1-i], re.IGNORECASE):
                if len(slice_1) > i + 1:
                    conclusion = slice_1[-2-i]
                    break
    
    return abstract, introduction, conclusion


def extract_entities(llm: LLMClient, abstract: str, intro: str, conclusion: str) -> Tuple[List[str], int]:
    """Stage 3: Extract entities using LLM"""
    prompt = f"""
## Generation Requirements:
1. Please extract 15 entities from the above text.
2. The format of the generation is as follows:
["Entity 1","Entity 2",...]
3. Do not generate any text that is not related to the generation requirements.
4. The extracted entities only include academic entities such as concepts and academic terms, and do not include entities that are not related to academics, such as names, dates, and places.
Abstract: {abstract or "None"}
Introduction: {intro or "None"}
Conclusion: {conclusion or "None"}
"""
    response, tokens = llm.chat(prompt)
    
    try:
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1
        entities = json.loads(response[start_idx:end_idx])
    except:
        entities = []
    
    return entities, tokens


def generate_questions(llm: LLMClient, entities: List[str]) -> Tuple[List[str], int]:
    """Stage 4: Generate questions from entities"""
    prompt = f"""\
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
   - Application (How to apply)
   - Historical development (Evolution process)
   - Mathematical principles (How it works/calculates)

## Format Requirements:
["Question 1","Question 2",...,"Question 10"]
## Entity List:
{chr(10).join([f"- {e}" for e in entities])}
Output ONLY the JSON-formatted list of questions that meet the requirements, without any explanations.
"""
    
    max_retries = 3
    for attempt in range(max_retries):
        response, tokens = llm.chat(prompt)
        
        try:
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            questions = json.loads(response[start_idx:end_idx])
            
            if len(questions) >= 5:
                cleaned = [
                    q.replace("In this paper", "").replace("In this study", "")
                    .replace("According to the paper", "").replace("As mentioned in the research", "")
                    .replace("the authors'", "").replace("the proposed", "")
                    .strip(' ,.;')
                    for q in questions
                ]
                return cleaned[:10], tokens
        except:
            pass
    
    # Fallback
    templates = cycle([
        "What is the scientific definition of {entity}?",
        "How does {entity} fundamentally operate?",
        "What are the key components of {entity}?",
        "What problem space does {entity} primarily address?",
        "What are the limitations of {entity}?"
    ])
    selected = entities[:10] if len(entities) >= 10 else (entities * (10 // len(entities) + 1))[:10]
    return [next(templates).format(entity=e) for e in selected], 0


# ============== Report Generator ==============
def generate_report(result: BenchmarkResult, output_path: str):
    """Generate markdown benchmark report"""
    
    report = f"""# QAG_System End-to-End Benchmark

**Generated at:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Input PDF | `{result.pdf_name}` |
| LLM Model | `{result.config.model}` |
| Embedding Model | `{result.config.embedding_model}` |
| Reranker Model | `{result.config.reranker_model}` |
| Temperature | {result.config.temperature} |

---

## 2. Overall Summary

| Metric | Value |
|--------|-------|
| **Total Duration** | {result.total_duration:.2f} seconds |
| **Total Tokens Used** | {result.total_tokens:,} |
| **Peak GPU Memory** | {result.peak_gpu_memory_mb:.2f} MB |
| **Entities Extracted** | {len(result.entities)} |
| **Questions Generated** | {len(result.questions)} |
| **QA Pairs Created** | {len(result.qa_pairs)} |

---

## 3. Stage-by-Stage Breakdown

| Stage | Duration (s) | Tokens | GPU Mem (MB) | CPU % | RAM (MB) |
|-------|-------------|--------|--------------|-------|----------|
"""
    
    for stage in result.stages:
        status = "✅" if not stage.error else "❌"
        report += f"| {status} {stage.name} | {stage.duration:.2f} | {stage.tokens_used} | {stage.gpu_memory_mb:.1f} | {stage.cpu_percent:.1f} | {stage.ram_used_mb:.1f} |\n"
    
    report += f"""
---

## 4. Timeline Visualization

```
"""
    
    # Simple ASCII timeline
    if not result.stages:
        report += "No stages completed.\n"
    else:
        max_name_len = max(len(s.name) for s in result.stages)
        max_duration = max(s.duration for s in result.stages) if result.stages else 1
        bar_width = 40
        
        for stage in result.stages:
            bar_len = int((stage.duration / max_duration) * bar_width) if max_duration > 0 else 0
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            report += f"{stage.name.ljust(max_name_len)} |{bar}| {stage.duration:.2f}s\n"
    
    report += f"""```

---

## 5. Extracted Entities

"""
    for i, entity in enumerate(result.entities, 1):
        report += f"{i}. {entity}\n"
    
    report += f"""
---

## 6. Generated Questions

"""
    for i, q in enumerate(result.questions, 1):
        report += f"{i}. {q}\n"
    
    report += f"""
---

## 7. QA Pairs (Alpaca Format)

```json
[
"""
    
    alpaca_entries = []
    for qa in result.qa_pairs:
        entry = {
            "instruction": qa["question"],
            "input": "",
            "output": qa["answer"],
            "system": "You are a helpful academic assistant."
        }
        alpaca_entries.append(json.dumps(entry, ensure_ascii=False, indent=2))
    
    report += ",\n".join(alpaca_entries)
    report += f"""
]
```

---

## 8. System Information

| Metric | Value |
|--------|-------|
| Python Version | {sys.version.split()[0]} |
| PyTorch Version | {torch.__version__} |
| CUDA Available | {torch.cuda.is_available()} |
| GPU Device | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"} |
| Total GPU Memory | {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB |

---

*Report generated by QAG_System Benchmark Tool*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Also save raw QA pairs as JSONL
    jsonl_path = output_path.replace('.md', '_alpaca.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for qa in result.qa_pairs:
            entry = {
                "instruction": qa["question"],
                "input": "",
                "output": qa["answer"],
                "system": "You are a helpful academic assistant."
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    return report


# ============== Main Benchmark ==============
def run_benchmark(config_path: str = "config.json"):
    """Run the complete benchmark pipeline"""
    
    print("=" * 60)
    print("QAG System Benchmark")
    print("=" * 60)
    
    # Load config
    config = BenchmarkConfig.from_json(config_path)
    result = BenchmarkResult(config=config)
    result.pdf_name = Path(config.input_pdf).name
    result.total_start_time = time.time()
    
    # Initialize LLM client
    llm = LLMClient(config)
    
    # Output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # ========== Stage 1: PDF to Markdown ==========
        print("\n[1/5] Converting PDF to Markdown...")
        stage1 = StageMetrics(name="PDF to Markdown")
        stage1.start()
        
        md_path = convert_pdf_to_markdown(
            config.input_pdf, 
            str(output_dir / "markdown")
        )
        
        stage1.stop()
        stage1.output_data = md_path
        result.add_stage(stage1)
        print(f"      ✅ Done in {stage1.duration:.2f}s -> {md_path}")
        
        # Clear MinerU models from GPU memory
        print("      🧹 Clearing GPU memory...")
        clear_gpu_memory()
        
        # ========== Stage 2: Extract Sections ==========
        print("\n[2/5] Extracting Abstract/Introduction/Conclusion...")
        stage2 = StageMetrics(name="Section Extraction")
        stage2.start()
        
        abstract, intro, conclusion = extract_sections(md_path)
        
        stage2.stop()
        stage2.output_data = {"abstract": bool(abstract), "intro": bool(intro), "conclusion": bool(conclusion)}
        result.add_stage(stage2)
        print(f"      ✅ Done in {stage2.duration:.2f}s")
        print(f"         Abstract: {'Found' if abstract else 'Not found'}")
        print(f"         Introduction: {'Found' if intro else 'Not found'}")
        print(f"         Conclusion: {'Found' if conclusion else 'Not found'}")
        
        # ========== Stage 3: Entity Extraction ==========
        print("\n[3/5] Extracting Entities using LLM...")
        stage3 = StageMetrics(name="Entity Extraction")
        stage3.start()
        
        entities, tokens3 = extract_entities(llm, abstract, intro, conclusion)
        
        stage3.stop(tokens3)
        stage3.output_data = entities
        result.entities = entities
        result.add_stage(stage3)
        print(f"      ✅ Done in {stage3.duration:.2f}s, {tokens3} tokens")
        print(f"         Extracted {len(entities)} entities")
        
        # ========== Stage 4: Question Generation ==========
        print("\n[4/5] Generating Questions...")
        stage4 = StageMetrics(name="Question Generation")
        stage4.start()
        
        questions, tokens4 = generate_questions(llm, entities)
        
        stage4.stop(tokens4)
        stage4.output_data = questions
        result.questions = questions
        result.add_stage(stage4)
        print(f"      ✅ Done in {stage4.duration:.2f}s, {tokens4} tokens")
        print(f"         Generated {len(questions)} questions")
        
        # Clear memory before loading RAG models
        print("      🧹 Clearing GPU memory before RAG...")
        clear_gpu_memory()
        
        # ========== Stage 5: Answer Generation (Mini_RAG) ==========
        print("\n[5/5] Generating Answers using Mini_RAG...")
        stage5 = StageMetrics(name="Answer Generation (Mini_RAG)")
        stage5.start()
        
        # Process document for RAG (reuse embedding model)
        processor = DocumentProcessor(config)
        chunks = processor.process_documents(md_path)
        
        # Create vector DB path
        vector_db_path = str(output_dir / "vector_db")
        
        # Initialize Mini_RAG (reuse embedding model from processor)
        rag = MiniRAG(chunks, vector_db_path, config, llm, embed_model=processor.embed_model)
        
        # Generate answers
        qa_pairs = []
        total_tokens5 = 0
        
        for i, question in enumerate(questions, 1):
            print(f"      Processing Q{i}/{len(questions)}...", end="\r")
            qa_result, tokens = rag.ask(question)
            qa_pairs.append(qa_result)
            total_tokens5 += tokens
        
        stage5.stop(total_tokens5)
        stage5.output_data = qa_pairs
        result.qa_pairs = qa_pairs
        result.add_stage(stage5)
        print(f"      ✅ Done in {stage5.duration:.2f}s, {total_tokens5} tokens")
        print(f"         Generated {len(qa_pairs)} QA pairs")
        
        # Final cleanup
        del rag, processor, chunks
        clear_gpu_memory()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Calculate totals
    result.total_end_time = time.time()
    result.total_duration = result.total_end_time - result.total_start_time
    
    # Generate report
    print("\n" + "=" * 60)
    print("Generating Report...")
    report_path = str(output_dir / "report.md")
    generate_report(result, report_path)
    print(f"✅ Report saved to: {report_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Duration:     {result.total_duration:.2f} seconds")
    print(f"Total Tokens:       {result.total_tokens:,}")
    print(f"Peak GPU Memory:    {result.peak_gpu_memory_mb:.2f} MB")
    print(f"Entities:           {len(result.entities)}")
    print(f"Questions:          {len(result.questions)}")
    print(f"QA Pairs:           {len(result.qa_pairs)}")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    # Get config path from command line or use default
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    run_benchmark(config_path)
