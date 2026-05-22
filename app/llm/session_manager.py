import threading
from collections import defaultdict
from typing import List, Dict
from app.llm.memory_manager import load_recent_messages, KEEP_RECENT_ROUNDS
from app.llm.context_compact import smart_compact
from app.llm.context_optimizer import optimize_tool_results_for_llm, optimize_thinking_for_llm


class SessionManager:
    """
    会话管理器 - 管理多个会话的消息列表

    功能：
    1. 线程安全的消息存储和访问
    2. 自动从 memory 加载历史记录
    3. 加载时自动应用上下文优化（tool_result 压缩、thinking 删除）
    4. 支持可选的强制上下文压缩

    每个 session 的消息按时间顺序存储，格式兼容 Anthropic SDK

    Attributes:
        messages: Dict[str, List], session_id 到消息列表的映射
        _loaded_from_memory: Set[str], 记录已从持久化存储加载的 session_id
        _lock: threading.Lock, 用于线程安全的锁
    """

    def __init__(self):
        self.messages: Dict[str, List] = {}
        self._loaded_from_memory: set = set()
        self._lock = threading.Lock()

    def get_messages(
            self,
            session_id: str,
            rounds: int = KEEP_RECENT_ROUNDS,
            force_compact: bool = False
    ) -> List[Dict]:
        """
        线程安全地获取或创建消息列表

        如果 session 不存在，会先从 memory 加载历史记录（如果有的话）。
        加载时会自动应用：
        1. tool_result 压缩（旧轮次的大结果会被截断）
        2. thinking 块删除（旧轮次的思考块会被移除）
        3. 可选的上下文压缩（如果 force_compact=True 且 rounds 较大）

        Args:
            session_id: 会话唯一标识符
            rounds: 要加载的历史轮数（默认从配置读取）
            force_compact: 是否强制进行上下文压缩

        Returns:
            List[Dict]: 该 session 的消息列表，格式为 Anthropic SDK 兼容的格式

        Note:
            使用双重检查锁（Double-Checked Locking）优化性能

        Example:
            >>> msgs = session.get_messages("user_123")
            >>> print(f"加载了 {len(msgs)} 条消息")
        """
        # 双重检查锁，提升性能
        if session_id not in self.messages:
            with self._lock:
                if session_id not in self.messages:
                    # 从 memory 加载历史记录
                    loaded = load_recent_messages(session_id, rounds=rounds)
                    loaded_len = len(loaded)

                    # 加载完后, 先处理 tool_result
                    loaded = optimize_tool_results_for_llm(loaded)

                    # 删除历史 thinking 块
                    loaded = optimize_thinking_for_llm(loaded)

                    # 如果 rounds > KEEP_RECENT_ROUNDS, 触发一次上下文压缩
                    if force_compact and rounds > KEEP_RECENT_ROUNDS:
                        loaded = smart_compact(loaded, session_id=session_id, force=True)

                    self.messages[session_id] = loaded
                    if loaded:
                        self._loaded_from_memory.add(session_id)
                        from app.log.logger import LOG
                        LOG.info(f"Session {session_id} 已从 memory 加载 {loaded_len} 条历史消息，"
                                f"最终加载 {len(loaded)} 条消息(可能触发了压缩)")
        return self.messages[session_id]

    def is_loaded_from_memory(self, session_id: str) -> bool:
        """
        检查指定 session 是否已从持久化存储加载过

        Args:
            session_id: 会话唯一标识符

        Returns:
            bool: True - 已从 memory 加载，False - 未加载或 session 不存在
        """
        return session_id in self._loaded_from_memory

    def clear_session(self, session_id: str):
        """
        清除指定 session 的所有消息

        从内存中删除该 session 的消息列表（不影响持久化存储）

        Args:
            session_id: 会话唯一标识符
        """
        with self._lock:
            if session_id in self.messages:
                del self.messages[session_id]

    def get_active_sessions(self) -> int:
        """
        获取当前活跃的 session 数量

        Returns:
            int: 当前在内存中缓存的 session 数量
        """
        with self._lock:
            return len(self.messages)


# 全局单例实例
session = SessionManager()