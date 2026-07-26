from pymongo import MongoClient

# 连接 MongoDB
client = MongoClient("mongodb://admin:123456@172.19.147.173:27017")
print(client)

# 选择数据库（不存在则自动创建）
db = client["mydb"]

# 选择集合（不存在则自动创建）
collection = db["students"]

# 更新单条
result = collection.update_one(
    {"name": "张三"},           # 查询条件
    {"$set": {"age": 21}}       # 更新操作
)
print(f"匹配 {result.matched_count} 条，修改 {result.modified_count} 条")

# 更新多条
result = collection.update_many(
    {"major": "计算机科学"},
    {"$set": {"status": "在读"}}
)
print(f"修改 {result.modified_count} 条")
