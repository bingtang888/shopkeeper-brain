from FlagEmbedding import FlagReranker

reranker = FlagReranker(
    model_name_or_path="D:\\ai_models\\modelscope_cache\\models\\BAAI\\BAAI\\bge-reranker-large",
    device="cpu",       # CPU 推理（无独显）
    use_fp16=False      # 全精度推理
)

# 计算相关性得分
pairs = [
    ["什么是万用表？", "万用表是一种测量电压、电流、电阻的仪器"],
    ["什么是万用表？", "今天天气很好"]
]
scores = reranker.compute_score(pairs)
print(f"scores: {scores}")
# 输出: [0.9234, 0.0156]  高分 = 高相关