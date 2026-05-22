#!/usr/bin/env python3
"""
快速嵌入模块 - 针对 MacBook Pro 优化

优化策略：
1. 使用轻量级模型（默认 all-MiniLM-L6-v2，仅22MB）
2. 支持 ONNX Runtime 加速
3. 支持 Apple Silicon MPS 后端
4. 模型单例缓存
"""
import os
import time
from typing import List, Optional
from functools import lru_cache

import numpy as np
from pathlib import Path
from app.log.logger import LOG

# 尝试导入加速器
try:
    import torch
    TORCH_AVAILABLE = True
    # 检查 MPS (Apple Silicon)
    MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

# 默认使用更轻更快的模型
DEFAULT_FAST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 22MB，384维
DEFAULT_CHINESE_MODEL = "BAAI/bge-small-zh-v1.5"  # 100MB+，512维

# 模型缓存目录
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR"))


class FastEmbedding:
    """
    快速嵌入模型管理器 - 单例模式

    针对 MacBook Pro 优化的 Embedding 模型管理器：
    1. 使用轻量级模型（默认 all-MiniLM-L6-v2，仅22MB）
    2. 支持本地缓存（首次下载后缓存到本地）
    3. 自动选择最优设备（默认 CPU 保证稳定性）
    4. 单例模式避免重复加载

    Attributes:
        model_name: str, 使用的模型名称
        device: str, 计算设备（"cpu" 或 "mps"）
        cache_dir: Path, 模型缓存目录

    Configuration:
        - EMBEDDING_MODEL: 指定模型名称
        - EMBEDDING_FORCE_CPU: 强制使用 CPU（默认 true）
        - MODEL_CACHE_DIR: 模型缓存目录

    Example:
        >>> model = FastEmbedding()
        >>> embeddings = model.encode(["文本1", "文本2"])
        >>> print(embeddings.shape)
        (2, 384)
    """

    _instance = None
    _model = None
    _model_name = None

    def __new__(cls):
        """单例模式：确保只有一个实例"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化模型（模型未加载时）"""
        if self._model is not None:
            return

        self.model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_FAST_MODEL)
        # MacBook 上 CPU 加载通常比 MPS 更稳定且速度差异不大
        self.device = self._get_optimal_device()
        self.cache_dir = MODEL_CACHE_DIR
        self._load_model()

    def _get_optimal_device(self) -> str:
        """
        获取最优计算设备

        注意：对于嵌入模型，CPU 通常比 MPS 更稳定且速度差异不大
        因为 MPS 初次加载有额外的编译开销
        """
        # 检查是否强制使用 CPU（推荐用于稳定性）
        force_cpu = os.getenv("EMBEDDING_FORCE_CPU", "true").lower() == "true"

        if force_cpu:
            LOG.info("使用 CPU 加载模型（稳定性优先）")
            return "cpu"

        if MPS_AVAILABLE:
            LOG.info("✅ 检测到 Apple Silicon MPS，使用 Metal 加速")
            return "mps"
        return "cpu"

    def _load_model(self):
        """加载模型（带计时和缓存优化）"""
        if not ST_AVAILABLE:
            raise RuntimeError("sentence-transformers 未安装")

        start_time = time.time()
        LOG.info(f"🚀 正在加载嵌入模型: {self.model_name}")
        LOG.info(f"   设备: {self.device}")
        LOG.info(f"   缓存目录: {self.cache_dir}")

        # 确保缓存目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 方法1: 尝试从本地缓存加载（最快）
            local_path = self.cache_dir / self.model_name.replace("/", "_")
            if local_path.exists():
                LOG.info(f"   从本地缓存加载: {local_path}")
                self._model = SentenceTransformer(
                    str(local_path),
                    device=self.device,
                    cache_folder=str(self.cache_dir)
                )
            else:
                # 方法2: 从 Hugging Face 下载并保存到本地
                LOG.info(f"   从 Hugging Face 下载模型...")
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=str(self.cache_dir)
                )
                # 保存到本地缓存以便下次快速加载
                try:
                    self._model.save(str(local_path))
                    LOG.info(f"   模型已缓存到: {local_path}")
                except Exception as e:
                    LOG.warning(f"   缓存保存失败: {e}")

            # 如果是 MPS，优化设置
            if self.device == "mps":
                if hasattr(torch, 'mps'):
                    torch.mps.empty_cache()

            load_time = time.time() - start_time
            LOG.info(f"✅ 模型加载完成: {load_time:.2f}s")
            LOG.info(f"   维度: {self._model.get_sentence_embedding_dimension()}")
            LOG.info(f"   模型大小: ~{self._estimate_model_size():.0f}MB")

        except Exception as e:
            LOG.error(f"❌ 模型加载失败: {e}，回退到 CPU")
            self._model = SentenceTransformer(self.model_name, device="cpu", cache_folder=str(self.cache_dir))

    def _estimate_model_size(self) -> float:
        """估算模型大小"""
        try:
            # 简单估算：参数数量 * 4字节 (float32)
            dim = self._model.get_sentence_embedding_dimension()
            # MiniLM-L6 约 22M 参数
            if "MiniLM" in self.model_name:
                return 22
            elif "bge-small" in self.model_name:
                return 100
            return dim * 0.1  # 粗略估算
        except:
            return 0

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        编码文本

        Args:
            texts: 文本列表
            normalize: 是否归一化

        Returns:
            嵌入向量
        """
        if self._model is None:
            raise RuntimeError("模型未加载")

        start = time.time()
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=32  # 批量处理提高效率
        )
        elapsed = time.time() - start

        if len(texts) > 0:
            LOG.debug(f"编码 {len(texts)} 条文本: {elapsed*1000:.1f}ms")

        return embeddings

    @property
    def dimension(self) -> int:
        """获取向量维度"""
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 0


# 全局实例（懒加载）
_fast_embedding: Optional[FastEmbedding] = None


def get_embedding_model() -> FastEmbedding:
    """获取嵌入模型实例（单例）"""
    global _fast_embedding
    if _fast_embedding is None:
        _fast_embedding = FastEmbedding()
    return _fast_embedding


def warm_up():
    """预热模型（首次编码较慢）"""
    model = get_embedding_model()
    warm_up_texts = ["预热", "warm up", "test"]
    _ = model.encode(warm_up_texts)
    LOG.info("模型预热完成")


# 兼容性函数，替换 memory_retrieval 中的调用
def encode_texts(texts: List[str]) -> np.ndarray:
    """外部调用接口"""
    model = get_embedding_model()
    return model.encode(texts)


if __name__ == "__main__":
    # 测试加载速度
    print("测试嵌入模型加载速度...")

    for model_name in [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-zh-v1.5"
    ]:
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name}")

        start = time.time()
        try:
            model = SentenceTransformer(model_name, device="mps" if MPS_AVAILABLE else "cpu")
            load_time = time.time() - start

            # 测试推理速度
            test_texts = ["这是一个测试句子"] * 10
            enc_start = time.time()
            _ = model.encode(test_texts, show_progress_bar=False)
            enc_time = time.time() - enc_start

            print(f"加载时间: {load_time:.2f}s")
            print(f"10条编码: {enc_time*1000:.1f}ms")
            print(f"维度: {model.get_sentence_embedding_dimension()}")

        except Exception as e:
            print(f"错误: {e}")
