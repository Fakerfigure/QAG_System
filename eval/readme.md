## 排名
| 模型                                          | 相关性          | 不可知性                   | 完整性                    | 准确性               | 合理性                   |
| ------------------------------------------- | ------------ | ---------------------- | ---------------------- | ----------------- | --------------------- |
| **mini_rag**                                | 0.603632     | **0.8804114975928712** | **0.6871870833112654** | 0.693563218390804 | **0.848793103448274** |
| **qag_llm_7B**                              | 0.603219     | 0.6802734231520254     | 0.6440896965401429     | 0.519183673469387 | 0.823749999999998     |
| qwen2.5-7b-instruct-1m                      | **0.604325** | 0.7292116785548945     | 0.6577766042807041     | 0.568313492063492 | 0.818499999999997     |
| **qag_llm_14B** (输出结构较不稳定)                  | 0.582328     | 0.746567096712705      | 0.6345581306761997     | 0.458420944135229 | 0.829653679653676     |
| qwen_max (输出结构不稳定)                          | 0.601384     | 0.7417386554392096     | 0.669950580535213      | 0.634111268269684 | 0.799174917491747     |
| qwen_plus                                   | 0.595526     | 0.7724686983307729     | 0.6484162963304279     | 0.548535714285714 | 0.830999999999997     |
| deepseek-R1                                 | 0.589498     | 0.8712768756489216     | 0.6200208008623888     | 0.452238095238095 | 0.831999999999997     |
| deepseek-v3                                 | 0.597217     | 0.7942775760342797     | 0.6623800343261154     | 0.627900793650793 | 0.833499999999997     |
| deepseek-r1-distill-qwen-7b (输出结构**严重**不稳定) | 0.577169     | 0.5718479993948602     | 0.6038385910154815     | 0.370576923076923 | 0.808076923076921     |
| deepseek-r1-distill-qwen-14b                | 0.587628     | 0.8008830973384303     | 0.6396732166559109     | 0.537001322751323 | 0.83183333333333      |
| llama3.3-70b-instruct                       | 0.594519     | 0.7531757461611749     | 0.6566711994020092     | 0.579174603174603 | 0.819833333333331     |
| qwq_32b                                     | 0.591557     | 0.8316831009312385     | 0.6296227918092777     | 0.511341269841270 | 0.83833333333333      |

## 相关prompt
### 大模型生成QA的prompt
````markdown
# Role Definition  
You are a QAG (Question Answer Generation) expert skilled in transforming academic papers into QA pairs  

# Workflow  
Step 1. Extract 10 key entities from the article  
Step 2. Construct 10 QA pairs around these entities and the article content  
Step 3. Strictly follow the output format and only output QA pairs  

## QA Generation Principles  
1. Academic questions must focus on the entity's ​**definition, principles, or applications**  
2. Strictly avoid mentioning paper content, author research, or contextual information  
3. Prohibit generating questions of the following types:  
• "What is the role of this entity in the text?"  
• "How did the authors apply this entity?"  
• "Where does this entity appear in the paper?"  
4. Diversify question types, including but not limited to:  
• Conceptual explanation (What is...)  
• Technical comparison (Difference between... and...)  
• Application scenarios (How is it applied...)  
• Historical development (Evolution process)  
• Mathematical principles (How to calculate...)  

# Output Principles
1. Strictly follow the output format and only output QA pairs
2. Don't add any emphatic marks, any headings, any explanatory statements

# Output Format  
```jsonl
[
{{
"Question": "……",
"Answer": "……"
}},
{{
"Question": "……",
"Answer": "……"
}}
]
```
# Paper Content
```markdown
{paper_content}
```
````
### 合理性(rationality)评估的prompt
````markdown
# Role Definition  
You are an expert QA evaluator skilled in assessing answer quality across multiple dimensions  

# Workflow
1. Receive input: <Question> and <Answer>  
2. Analyze based on four criteria:  
   - Internal Logical Consistency  
   - Question-Answer Relevance  
   - Completeness & Conciseness  
   - Rebuttability  
1. Assign 0-5 scores per dimension (5=excellent)  
2. Provide brief rationale for each score  

# Definitions of each dimension
1. Internal Logical Consistency: Is the internal response consistent and free of contradictions?
2. Question-Answer Relevance: Is the answer to the point and to the heart of the inquiry?
3. Completeness & Conciseness: Is it neither redundant nor missing points?
4. Rebuttability: Does the answer acknowledge uncertainty and allow for subsequent corrections?

# Evaluation Criteria  
**1. Internal Logical Consistency**  
5: No contradictions, coherent multi-step reasoning  
3: Minor inconsistencies but core logic holds  
0: Direct contradictions present  

**2. Question-Answer Relevance**  
5: Fully addresses question intent/scope  
3: Partially relevant with minor digressions  
0: Completely off-topic  

**3. Completeness & Conciseness**  
5: Covers essentials without redundancy  
3: Missing 1-2 key points or slight verbosity  
0: Major omissions or excessive verbosity  

**4. Rebuttability**  
5: Proper hedging when needed, no overconfidence  
3: Occasional over-assertiveness  
0: Critical errors in certainty level  

# Output Principles
1. Strictly follow the output format
2. Don't add any emphatic marks, any headings, any explanatory statements

# Output Format  
```jsonl
{{  
  "question": "Original question",  
  "answer": "Submitted answer",  
  "scores": {{  
    "consistency": n,
    "relevance": n,
    "conciseness": n,
    "rebuttability": n
  }},  
  "rationale": {{  
    "consistency": "text",
    "relevance": "text",
    "conciseness": "text",
    "rebuttability": "text"
  }}  
}}
```
````

