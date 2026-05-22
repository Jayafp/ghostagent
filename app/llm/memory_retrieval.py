#!/usr/bin/env python3
"""
Memory Retrieval Module - 轻量级混合检索

检索流程：
1. BM25 关键词检索（带阈值过滤）
2. 向量检索（带阈值过滤）
3. RRF 融合排序
4. 动态阈值过滤（至少满足一种检索的阈值）
5. MMR 重排序（增加多样性）

优化点：
- 索引所有消息类型（user, assistant, tool_result）
- 查询意图扩展（关键词提取）
- RRF 融合保证综合质量
- MMR 重排序增加多样性，避免内容过于集中

依赖：
    pip install rank-bm25 sentence-transformers numpy jieba
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from app.log.logger import LOG

# 可选的BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

# 可选的中文分词
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False

# 可选的向量模型 - 使用优化版本
try:
    from app.llm.fast_embedding import get_embedding_model, FastEmbedding
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

from app.llm.memory_manager import MEMORY_DIR, list_session_memories


@dataclass
class SearchResult:
    """
    检索结果数据类

    存储单条检索结果的完整信息，包括内容、相关性分数和元信息

    Attributes:
        content: str, 消息内容的文本表示
        role: str, 消息角色（"user"/"assistant"）
        timestamp: str, 消息时间戳（ISO 格式）
        message_index: int, 在原始消息列表中的位置
        bm25_score: float, BM25 关键词检索分数（0.0-∞，越大越好）
        vector_score: float, 向量相似度分数（0.0-1.0，越大越相似）
        final_score: float, RRF 融合后的最终分数
    """
    content: str
    role: str
    timestamp: str
    message_index: int = 0
    bm25_score: float = 0.0
    vector_score: float = 0.0
    final_score: float = 0.0


# ============ 可配置的索引过滤选项 ============
# 从环境变量读取，默认包含所有
# MEMORY_INDEX_ROLES: user,assistant (逗号分隔)
# MEMORY_INDEX_TYPES: text,thinking,tool_use,tool_result (逗号分隔)

DEFAULT_INDEX_ROLES = {"user", "assistant"}


def _parse_env_set(env_name: str, default_set: set) -> set:
    """
    从环境变量解析逗号分隔的字符串为集合

    Args:
        env_name: 环境变量名
        default_set: 默认值（当环境变量未设置或为空时使用）

    Returns:
        set: 解析后的字符串集合

    Example:
        >>> # 环境变量 MEMORY_INDEX_ROLES="user,assistant"
        >>> _parse_env_set("MEMORY_INDEX_ROLES", {"user"})
        {"user", "assistant"}
    """
    env_value = os.getenv(env_name, "").strip()
    if not env_value:
        return default_set
    return set(item.strip() for item in env_value.split(",") if item.strip())


class MemoryRetriever:
    """
    记忆检索器 - 实现混合检索（BM25 + 向量检索）

    支持的功能：
    1. 从 memory 文件加载会话历史
    2. 构建 BM25 关键词索引
    3. 构建向量语义索引
    4. RRF 融合排序
    5. MMR 多样性重排序
    6. 动态阈值过滤

    Attributes:
        model_name: str, 使用的嵌入模型名称
        model: FastEmbedding, 向量模型实例
        bm25: BM25Okapi, BM25 索引实例
        documents: List[Dict], 索引的文档列表
        embeddings: np.ndarray, 向量嵌入矩阵
        _initialized: bool, 是否已初始化
        vector_threshold: float, 向量相似度阈值
        bm25_threshold: float, BM25 分数阈值
        index_roles: Set[str], 需要索引的消息角色

    Configuration:
        - EMBEDDING_MODEL: 嵌入模型名称
        - MEMORY_VECTOR_THRESHOLD: 向量相似度阈值（默认 0.3）
        - MEMORY_BM25_THRESHOLD: BM25 分数阈值（默认 0.5）
        - MEMORY_INDEX_ROLES: 索引的角色，逗号分隔（默认 "user,assistant"）
    """

    # 默认使用更快的轻量级模型
    DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    def __init__(self, model_name: Optional[str] = None):
        """
        初始化记忆检索器

        Args:
            model_name: 嵌入模型名称，默认从环境变量 EMBEDDING_MODEL 读取
        """
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", self.DEFAULT_MODEL)
        self.model = None
        self.bm25 = None
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self._initialized = False

        # 相似度阈值
        self.vector_threshold = float(os.getenv("MEMORY_VECTOR_THRESHOLD", "0.3"))
        self.bm25_threshold = float(os.getenv("MEMORY_BM25_THRESHOLD", "0.5"))

        # 索引过滤配置
        self.index_roles = _parse_env_set("MEMORY_INDEX_ROLES", DEFAULT_INDEX_ROLES)

        LOG.info(f"检索器配置 - 索引Roles: {self.index_roles}")

    def _load_model(self):
        """延迟加载向量模型（使用优化版本）"""
        if not ST_AVAILABLE or self.model is not None:
            return
        try:
            LOG.info(f"加载嵌入模型: {self.model_name}")
            # 使用单例模式获取模型，避免重复加载
            fast_emb = get_embedding_model()
            self.model = fast_emb
            self.model_name = fast_emb.model_name
            LOG.info(f"模型加载完成，维度: {fast_emb.dimension}")
        except Exception as e:
            LOG.exception(f"加载模型失败: {e}，向量检索将不可用")
            self.model = None

    def _tokenize(self, text: str) -> List[str]:
        """分词处理"""
        if not text:
            return []
        text = str(text).lower().strip()

        if JIEBA_AVAILABLE:
            # 使用 jieba 精确模式分词
            return list(jieba.cut(text, cut_all=False))
        else:
            # 简单字符分割（中英文混合）
            return re.findall(r'[\u4e00-\u9fff]|\w+', text)

    def _extract_keywords(self, text: str, top_k: int = 5) -> List[str]:
        """提取关键词（用于查询扩展）"""
        if not JIEBA_AVAILABLE or not text:
            return self._tokenize(text)
        try:
            keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=False)
            return keywords if keywords else self._tokenize(text)
        except:
            return self._tokenize(text)

    def _extract_all_content(self, record: Dict) -> Optional[str]:
        """
        从消息记录中提取完整可索引内容，应用角色和类型过滤

        Returns:
            提取的文本，如果该记录应被过滤则返回 None
        """
        role = record.get("role", "")

        # 1. 检查 role 是否在允许的索引范围内
        if role not in self.index_roles:
            return None

        content = record.get("content", "")
        texts = []

        # 2. 根据类型提取内容（选择性提取符合白名单的类型）
        if isinstance(content, str):
            # 纯文本消息，视为 "text" 类型
            texts.append(content)
        elif isinstance(content, list):
            text = None
            # 如果存在非 text 和 thinking 类型的块，整个内容过滤掉
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type", "")
                if block_type != 'text' and block_type != 'thinking':
                    text = None
                    break
                if block_type == "text":
                    text = block.get("text", "")
            if text:
                texts.append(text)

        full_text = " ".join(filter(None, texts))

        if len(full_text.strip()) < 3:  # 过滤太短的内容
            return None

        # 3. 添加角色标记，有助于语义区分
        if role == "user":
            return f"[用户问题] {full_text}"
        elif role == "assistant":
            return f"[助手回答] {full_text}"
        else:
            return full_text

    def load_session_memories(self, session_id: str) -> int:
        """从 memory 目录加载指定 session 的所有历史记录"""
        self.documents = []
        memory_files = list_session_memories(session_id)

        if not memory_files:
            LOG.info(f"未找到 session {session_id} 的历史记录")
            return 0

        # 统计信息
        total_records = 0
        filtered_records = 0
        role_filter_stats = {}
        type_filter_stats = {}

        doc_index = 0
        for file_path in memory_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        role = record.get("role", "")
                        content = self._extract_all_content(record)

                        # 过滤被配置排除的内容
                        if content is None:
                            # 记录被过滤，跳过
                            continue

                        # 过滤太短或无意义的内容
                        if len(content) < 3:
                            continue

                        self.documents.append({
                            "content": content,
                            "original_content": record.get("content", ""),
                            "role": role,
                            "timestamp": record.get("timestamp", ""),
                            "index": doc_index,
                            "source_file": str(file_path)
                        })
                        doc_index += 1

                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                LOG.warning(f"读取 memory 文件失败 [{file_path}]: {e}")

        LOG.info(f"从 session {session_id} 加载了 {len(self.documents)} 条记录")
        self._build_index()
        return len(self.documents)

    def _build_index(self):
        """构建检索索引"""
        if not self.documents:
            return

        # BM25 索引
        if BM25_AVAILABLE:
            tokenized_docs = [self._tokenize(doc["content"]) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            LOG.info(f"BM25 索引构建完成: {len(tokenized_docs)} 文档")

        # 向量索引
        if ST_AVAILABLE:
            self._load_model()
            if self.model:
                texts = [doc["content"] for doc in self.documents]
                LOG.info(f"正在编码 {len(texts)} 条文档到向量...")
                # 使用 FastEmbedding 的 encode 方法（自动归一化）
                self.embeddings = self.model.encode(texts, normalize=True)
                LOG.info(f"向量索引完成: {self.embeddings.shape}")

        self._initialized = True

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = None,
        vector_weight: float = None
    ) -> List[SearchResult]:
        """
        混合检索，严格遵循阈值过滤

        只有 BM25 分数和向量相似度都超过配置阈值的结果才会被召回

        Args:
            query: 用户查询
            top_k: 返回结果数量上限
            bm25_weight: (兼容旧版，不再使用)
            vector_weight: (兼容旧版，不再使用)
        """
        if not self._initialized or not self.documents:
            return []

        query = query.strip()
        if len(query) < 2:
            return []

        results_map: Dict[int, SearchResult] = {}

        # ========== 1. BM25 检索 ==========
        if self.bm25 and BM25_AVAILABLE:
            # 提取关键词进行查询扩展
            keywords = self._extract_keywords(query, top_k=5)
            expanded_query = " ".join(keywords) if keywords else query

            query_tokens = self._tokenize(expanded_query)
            bm25_scores = self.bm25.get_scores(query_tokens)

            # 收集超过阈值的结果（仅BM25自身的过滤）
            bm25_candidates = []
            for idx, score in enumerate(bm25_scores):
                if score > self.bm25_threshold:
                    bm25_candidates.append((idx, score))

            # 限制最多 top_k*4 个候选（给MMR更多选择）
            bm25_candidates = sorted(bm25_candidates, key=lambda x: x[1], reverse=True)[:top_k * 4]

            for idx, score in bm25_candidates:
                doc = self.documents[idx]
                # 使用过滤后的 content（纯文本）
                results_map[idx] = SearchResult(
                    content=doc["content"],
                    role=doc["role"],
                    timestamp=doc["timestamp"],
                    message_index=doc["index"],
                    bm25_score=float(score)
                )

        # ========== 2. 向量检索 ==========
        query_vec = None
        if self.embeddings is not None and ST_AVAILABLE and self.model:
            # 对查询进行编码并归一化
            query_vec = self.model.encode([query], normalize=True)[0]

            # 计算余弦相似度（已经归一化，直接点乘）
            similarities = np.dot(self.embeddings, query_vec)

            # 收集超过阈值的结果（仅向量自身的过滤）
            vec_candidates = []
            for idx, sim in enumerate(similarities):
                if sim > self.vector_threshold:
                    vec_candidates.append((idx, float(sim)))

            # 限制最多 top_k*4 个候选（给MMR更多选择）
            vec_candidates = sorted(vec_candidates, key=lambda x: x[1], reverse=True)[:top_k * 4]

            for idx, sim in vec_candidates:
                if idx in results_map:
                    results_map[idx].vector_score = sim
                else:
                    doc = self.documents[idx]
                    # 使用过滤后的 content（纯文本）
                    results_map[idx] = SearchResult(
                        content=doc["content"],
                        role=doc["role"],
                        timestamp=doc["timestamp"],
                        message_index=doc["index"],
                        vector_score=sim
                    )

        # ========== 3. RRF 融合排序 ==========
        results = self._rrf_fuse(list(results_map.values()), k=60)

        # todo 是否需要参考 openclaw, 对检索数据做根据时间的衰减? openclaw的做法:
        # 时间衰减公式：
        #  - 衰减系数 = e^(-λ × 天数)
        #  - 其中 λ = ln(2) / 半衰期天数（默认 30 天）

        # 过滤：只保留同时满足两个阈值的结果（更严格）
        qualified_results = [
            r for r in results
            if r.bm25_score > self.bm25_threshold and r.vector_score > self.vector_threshold
        ]

        # ========== 4. MMR 重排序（增加多样性） ==========
        # 如果有查询向量，使用 MMR 重排序增加多样性
        if query_vec is not None and len(qualified_results) > top_k:
            # 使用 RRF 排序后的前 K*2 个候选进行 MMR 重排序
            mmr_candidates = qualified_results[:top_k * 2]
            mmr_results = self._mmr_rerank(
                mmr_candidates,
                query_vec,
                top_k=top_k,
                lambda_param=0.5  # 平衡相关性和多样性
            )
            LOG.info(f"MMR：top_k: {top_k}，候选结果数量: {len(qualified_results)}，MMR 重排序后结果数: {len(mmr_results)}")
            return mmr_results

        # 向量模型不可用或结果不足，直接返回 RRF 结果
        return qualified_results[:top_k]

    def _rrf_fuse(self, results: List[SearchResult], k: int = 60) -> List[SearchResult]:
        """RRF 融合排序"""
        if not results:
            return []

        # 初始化 RRF 分数
        for result in results:
            result.final_score = 0.0

        # 按 BM25 分数排序并累加 RRF 分数
        bm25_ranked = sorted(
            [(i, r) for i, r in enumerate(results) if r.bm25_score > 0],
            key=lambda x: x[1].bm25_score, reverse=True
        )
        for rank, (idx, _) in enumerate(bm25_ranked):
            results[idx].final_score += 1.0 / (k + rank)

        # 按向量分数排序并累加 RRF 分数
        vec_ranked = sorted(
            [(i, r) for i, r in enumerate(results) if r.vector_score > 0],
            key=lambda x: x[1].vector_score, reverse=True
        )
        for rank, (idx, _) in enumerate(vec_ranked):
            results[idx].final_score += 1.0 / (k + rank)

        return sorted(results, key=lambda x: x.final_score, reverse=True)

    def _mmr_rerank(
        self,
        results: List[SearchResult],
        query_vec: np.ndarray,
        top_k: int = 5,
        lambda_param: float = 0.5
    ) -> List[SearchResult]:
        """
        MMR (Maximum Marginal Relevance) 重排序

        在 RRF 融合排序后，使用 MMR 在保持相关性的同时增加结果的多样性。
        这样可以避免返回的内容过于集中在某个时间段或某个对话轮次。

        公式: MMR = λ * Relevance(d) - (1-λ) * max(Sim(d, d_i))

        Args:
            results: RRF 融合排序后的候选结果（已通过阈值过滤）
            query_vec: 查询向量 (已归一化)
            top_k: 最终返回结果数
            lambda_param: 相关性vs多样性权衡参数
                         - λ=1: 只考虑相关性（退化为按 RRF 排序）
                         - λ=0: 只考虑多样性
                         - 默认 0.5: 平衡两者
        """
        if not results or self.embeddings is None:
            return results[:top_k]

        if len(results) <= top_k:
            return results

        # 获取候选文档的索引
        candidate_indices = [r.message_index for r in results]

        # 计算候选文档之间的相似度矩阵（使用向量嵌入）
        candidate_embeddings = self.embeddings[candidate_indices]

        # 计算候选与查询的相似度
        query_similarities = np.dot(candidate_embeddings, query_vec)

        # 预计算候选文档之间的所有相似度（避免循环中重复计算）
        pairwise_similarities = np.dot(candidate_embeddings, candidate_embeddings.T)

        # MMR 选择
        selected: List[int] = []
        remaining = set(range(len(results)))

        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = None

            for idx in remaining:
                # 相关性分数（使用 RRF 分数和向量相似度的加权）
                relevance = 0.6 * results[idx].final_score + 0.4 * query_similarities[idx]

                # 多样性惩罚：与已选文档的最大相似度
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = float(np.max(pairwise_similarities[idx, selected]))

                # MMR 分数
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity_penalty

                # 额外的多样性奖励：如果 role 不同，给予奖励
                if selected:
                    current_role = results[idx].role
                    selected_roles = {results[s].role for s in selected}
                    if current_role not in selected_roles:
                        mmr_score += 0.05  # 小奖励，鼓励 role 多样性

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        # 按选中顺序返回结果
        return [results[i] for i in selected]

    def format_for_prompt(self, results: List[SearchResult], max_length: int = 400) -> str:
        """格式化检索结果用于 prompt

        Args:
            results: 检索结果列表
            max_length: 每条内容的最大长度（兼容旧版参数，实际固定400字符）
        """
        if not results:
            return "(无相关历史信息)"

        # 不需要标题, 已经在system prompt里加了
        #lines = ["## 相关历史信息", ""]
        lines = []

        for i, r in enumerate(results, 1):
            # 提取纯文本内容
            content = self._format_content_for_display(r.content)
            content = content[:max_length] + "..." if len(content) > max_length else content

            role_icon = {"user": "👤", "assistant": "🤖"}.get(r.role, "📝")
            date = r.timestamp[:10] if r.timestamp else "历史"

            lines.append(f"{i}. {role_icon} [{date}] {content}")

        return "\n".join(lines)

    def _format_content_for_display(self, content) -> str:
        """将内容格式化为可读的纯文本"""
        if isinstance(content, str):
            return content.replace("\n", " ")

        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", ""))
                    elif block.get("type") == "tool_result":
                        texts.append(f"[结果: {str(block.get('content', ''))[:100]}]")
            return " ".join(texts)

        return str(content)


# 全局检索器缓存 {session_id:model_name -> MemoryRetriever}
_retriever_cache: Dict[str, MemoryRetriever] = {}


def get_retriever(
    session_id: str,
    model_name: Optional[str] = None,
    force_reload: bool = False
) -> Optional[MemoryRetriever]:
    """
    获取或创建检索器

    使用缓存机制避免重复加载，相同 session_id 和 model_name 的检索器会被复用

    Args:
        session_id: 会话唯一标识符
        model_name: 嵌入模型名称，None 使用默认
        force_reload: 是否强制重新加载（清除缓存）

    Returns:
        MemoryRetriever: 检索器实例
        None: 如果该 session 没有历史记录

    Example:
        >>> retriever = get_retriever("user_123")
        >>> results = retriever.search("查询内容")
    """
    cache_key = f"{session_id}:{model_name or 'default'}"

    if force_reload and cache_key in _retriever_cache:
        del _retriever_cache[cache_key]

    if cache_key not in _retriever_cache:
        retriever = MemoryRetriever(model_name)
        count = retriever.load_session_memories(session_id)
        if count > 0:
            _retriever_cache[cache_key] = retriever
        else:
            return None

    return _retriever_cache.get(cache_key)


def clear_retriever_cache(session_id: str = None) -> None:
    """
    清除检索器缓存

    Args:
        session_id: 会话ID，None 表示清除所有缓存

    Returns:
        None

    Note:
        当 session 数据发生变化（如新消息）时，需要清除对应缓存
    """
    global _retriever_cache
    if session_id is None:
        _retriever_cache.clear()
    else:
        keys = [k for k in _retriever_cache if k.startswith(f"{session_id}:")]
        for k in keys:
            del _retriever_cache[k]
