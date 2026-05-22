#!/usr/bin/env python3
"""
下载 BGE 模型脚本（支持国内镜像）
"""
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv(override=True)

# 优先使用 ModelScope 镜像（国内快）
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# 或者直接用 ModelScope
# os.environ.setdefault('MODELSCOPE_CACHE', './models')

MODEL_NAME = os.getenv('EMBEDDING_MODEL', "BAAI/bge-small-zh-v1.5")
SAVE_PATH = os.getenv('MODEL_CACHE_DIR') + "/bge-small-zh-v1.5"

print(f"正在下载模型: {MODEL_NAME}")
print(f"保存到: {SAVE_PATH}")

model = SentenceTransformer(MODEL_NAME, cache_folder='./models')
model.save(SAVE_PATH)

print(f"模型已保存到: {SAVE_PATH}")
print(f"请修改 .env: EMBEDDING_MODEL = \"{SAVE_PATH}\"")