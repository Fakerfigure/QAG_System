# 不可知性权重变化结果分析
我们进一步分析了评估方法对不可知性加参数的敏感性。通过在得分之间以不同权重组合（从[0.1, 0.9]至[0.9, 0.1]）计算综合评分，我们记录了各模型排名随权重变化的情况。
![test4](https://github.com/user-attachments/assets/8d90639c-4c26-4b7a-a97c-7add116ca0c6)
- 某些模型（如 deepseek-R1, qwq_32b）在参数变化下排名保持稳定，表明其性能在不同评价重点下均较为均衡；而其他模型如 qwen_max、llama3.3-70b-instruct 等在权重从偏向BLEU切换至偏向惩罚时排名大幅下降，表现出较强的参数敏感性。
- 不可知性排名：
model	[0.1,0.9]	[0.2,0.8]	[0.3,0.7]	[0.4,0.6]	[0.5,0.5]	[0.6,0.4]	[0.7,0.3]	[0.8,0.2]	[0.9,0.1]
mini_rag	7	5	2	2	1	1	1	1	1
qag_llm_7B	11	11	11	11	11	11	11	9	8
qwen2.5-7b-instruct-1m	9	9	10	10	10	10	8	8	9
qag_llm_14B	10	10	9	9	8	7	6	5	5
qwen_max	6	7	8	8	9	9	10	10	10
qwen_plus	8	8	7	7	6	6	7	7	7
deepseek-R1	1	1	1	1	2	2	2	2	2
deepseek-v3	2	2	4	5	5	5	5	6	6
deepseek-r1-distill-qwen-7b	12	12	12	12	12	12	12	12	12
deepseek-r1-distill-qwen-14b	4	4	5	4	4	4	4	4	4
llama3.3-70b-instruct	3	6	6	6	7	8	9	11	11
qwq_32b	5	3	3	3	3	3	3	3	3
