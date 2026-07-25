from pymilvus.model.hybrid import BGEM3EmbeddingFunction

# bge3-m3 嵌入模型默认输出的稠密向量维度是1024
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='D:\\ai_models\\modelscope_cache\\models\\BAAI\\bge-m3', # Specify the model name
    device='cpu', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=False # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)

# 嵌入对象 (query or document)

vector_result = bge_m3_ef.encode_queries(queries=["我是中国人","你是美国人"])  # 用户问题嵌入

# 1、稠密向量
print(vector_result.get('dense')[0].tolist())

# 2、稀疏向量(token_id:权重) ---> {"token_id":"权重","token_id":"权重","token_id":"权重"}  ---> Milvus用户稀疏向量的结构必须是一个字典且key:必须是token_id  value：必须是权重
sparse_array = vector_result.get('sparse')
print(sparse_array)
# bge_m3_ef.decode_documents() # 文档内容嵌入