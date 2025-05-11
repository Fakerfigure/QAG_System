## 相关prompt
### 大模型生成QA的prompt
```markdown
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
```
