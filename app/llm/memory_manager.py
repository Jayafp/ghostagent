#!/usr/bin/env python3
"""
Memory Manager - 持久化对话历史管理

目录结构: memory/{session_id}/{YYYY-MM-DD}.jsonl

每行 jsonl 格式:
{
    "timestamp": "2026-03-29T10:30:00",
    "role": "user/assistant",
    "content": "..." or [...]  # Anthropic SDK 兼容格式
}
"""
import os
import json
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.llm.utils import is_pure_user_message
from app.log.logger import LOG

# ============ 配置常量 ============
MEMORY_DIR = Path(os.getenv("MEMORY_DIR"))

# Keep recent rounds for memory recovery
KEEP_RECENT_ROUNDS = int(os.getenv("KEEP_RECENT_ROUNDS", "30"))
INIT_RECENT_ROUNDS = int(os.getenv("INIT_RECENT_ROUNDS", "100"))


def get_session_memory_dir(session_id: str) -> Path:
    """
    获取指定 session 的 memory 目录路径

    Args:
        session_id: 会话唯一标识符

    Returns:
        Path: memory 目录路径，格式为 {MEMORY_DIR}/{session_id}
    """
    return MEMORY_DIR / session_id


def get_today_memory_file(session_id: str) -> Path:
    """
    获取指定 session 今天的 memory 文件路径

    Args:
        session_id: 会话唯一标识符

    Returns:
        Path: 今日 memory 文件路径，格式为 {MEMORY_DIR}/{session_id}/{YYYY-MM-DD}.jsonl
    """
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return get_session_memory_dir(session_id) / f"{today}.jsonl"


def serialize_content(content: Any) -> Any:
    """
    序列化消息内容为可 JSON 存储的格式

    支持的内容类型：
    - 字符串：直接返回
    - 列表（Anthropic 内容块）：转换为字典列表
    - Pydantic 模型：调用 model_dump()
    - Anthropic 类型对象：提取关键字段转为字典

    Args:
        content: 原始消息内容

    Returns:
        Any: 可 JSON 序列化的内容
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = []
        for block in content:
            if isinstance(block, dict):
                result.append(block)
            elif hasattr(block, "model_dump"):
                # Pydantic 模型
                result.append(block.model_dump())
            elif hasattr(block, "type"):
                # Anthropic 类型对象
                result.append(_extract_block_dict(block))
            else:
                result.append({"type": "unknown", "data": str(block)})
        return result

    return str(content)


def _extract_block_dict(block: Any) -> Dict:
    """
    将 Anthropic 内容块转换为字典

    Args:
        block: Anthropic content block 对象（text/thinking/tool_use/tool_result）

    Returns:
        Dict: 包含以下字段的字典：
            - type: str, 块类型（"text"/"thinking"/"tool_use"/"tool_result"）
            - text: str, text 类型的内容（仅 type="text" 时存在）
            - thinking: str, thinking 类型的内容（仅 type="thinking" 时存在）
            - id: str, tool_use 的 id（仅 type="tool_use" 时存在）
            - name: str, tool_use 的工具名（仅 type="tool_use" 时存在）
            - input: dict, tool_use 的参数（仅 type="tool_use" 时存在）
            - tool_use_id: str, tool_result 的关联 id（仅 type="tool_result" 时存在）
            - content: str, tool_result 的内容（仅 type="tool_result" 时存在）
    """
    block_type = getattr(block, "type", "unknown")
    data = {"type": block_type}

    if block_type == "text":
        data["text"] = getattr(block, "text", "")
    elif block_type == "thinking":
        data["thinking"] = getattr(block, "thinking", "")
    elif block_type == "tool_use":
        data["id"] = getattr(block, "id", "")
        data["name"] = getattr(block, "name", "")
        data["input"] = getattr(block, "input", {})
    elif block_type == "tool_result":
        data["tool_use_id"] = getattr(block, "tool_use_id", "")
        data["content"] = getattr(block, "content", "")

    return data


def deserialize_content(content: Any) -> Any:
    """
    反序列化消息内容

    JSONL 存储为字典格式，Anthropic SDK 客户端可以直接处理字典格式，
    因此这里保持原样返回

    Args:
        content: JSON 解析后的内容

    Returns:
        Any: 内容对象（字典或列表）
    """
    # 保持原样，react_agent.py 中的 Anthropic 客户端可以处理字典格式
    return content


def append_message(session_id: str, role: str, content: Any) -> Optional[Path]:
    """
    追加单条消息到 memory 文件

    消息会被序列化为 JSON 格式并追加写入到当天的 jsonl 文件中

    Args:
        session_id: 会话唯一标识符
        role: 消息角色，可选值："user", "assistant"
        content: 消息内容（字符串或内容块列表）

    Returns:
        Path: 写入的文件路径
        None: 如果写入失败

    Note:
        文件格式为追加写入的 JSON Lines（.jsonl）
        每条记录包含：timestamp, role, content
    """
    try:
        # 确保目录存在
        memory_dir = get_session_memory_dir(session_id)
        memory_dir.mkdir(parents=True, exist_ok=True)

        # 准备记录
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "role": role,
            "content": serialize_content(content)
        }

        # 追加写入 jsonl
        memory_file = get_today_memory_file(session_id)
        with open(memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return memory_file

    except Exception as e:
        LOG.exception(f"写入 memory 失败 [session_id={session_id}]: {e}")
        return None


def append_round_messages(
    session_id: str,
    user_content: Any,
    assistant_content: Any,
    tool_results: Optional[List] = None
) -> Path:
    """
    追加一整轮对话到 memory

    一轮对话包含：
    1. user 消息
    2. assistant 消息
    3. tool_result 消息（如果有）

    Args:
        session_id: 会话唯一标识符
        user_content: 用户消息内容（tool_result 列表或字符串）
        assistant_content: 助手回复内容（content blocks 列表）
        tool_results: 工具调用结果列表（可选）

    Returns:
        Path: 写入的文件路径
    """
    # 写入 user 消息
    append_message(session_id, "user", user_content)

    # 写入 assistant 消息
    append_message(session_id, "assistant", assistant_content)

    # 如果有工具结果，作为 user 消息写入
    if tool_results:
        append_message(session_id, "user", tool_results)

    return get_today_memory_file(session_id)


def load_recent_messages(session_id: str, rounds: int = KEEP_RECENT_ROUNDS) -> List[Dict]:
    """
    从 memory 加载最近的对话记录

    加载策略：
    1. 查找 session 目录下的所有 .jsonl 文件
    2. 按时间顺序从新到旧读取
    3. 重建消息列表（保持时间顺序）
    4. 截取最近 rounds 轮的消息

    Args:
        session_id: 会话唯一标识符
        rounds: 要加载的轮数（每轮 = user + assistant + optional tool_result）

    Returns:
        List[Dict]: Anthropic SDK 格式的消息列表
        每个字典包含：{"role": str, "content": Any}

    Note:
        使用 is_pure_user_message() 判断新轮次的开始
    """
    try:
        memory_dir = get_session_memory_dir(session_id)
        if not memory_dir.exists():
            LOG.info(f"Memory 目录不存在 [session_id={session_id}]")
            return []

        # 获取所有 jsonl 文件，按日期排序（最新的在前）
        jsonl_files = sorted(memory_dir.glob("*.jsonl"), reverse=True)

        if not jsonl_files:
            LOG.info(f"Memory 文件不存在 [session_id={session_id}]")
            return []

        # 读取所有消息（从新到旧）
        all_messages = []
        for file_path in jsonl_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in reversed(f.readlines()):  # 从内到外，从最新到最旧
                        line = line.strip()
                        if not line:
                            continue
                        record = json.loads(line)
                        all_messages.insert(0, {  # 插入到开头保持时间顺序
                            "role": record["role"],
                            "content": deserialize_content(record["content"])
                        })
            except Exception as e:
                LOG.warning(f"读取 memory 文件失败 [{file_path}]: {e}")
                continue

        if not all_messages:
            return []

        # 按轮数截取（从后往前数 rounds 轮）
        round_count = 0
        split_index = 0

        for i in range(len(all_messages) - 1, -1, -1):
            if is_pure_user_message(all_messages[i]):
                round_count += 1
                if round_count == rounds:
                    split_index = i
                    break

        result = all_messages[split_index:]
        LOG.info(f"从 memory 加载了 {len(result)} 条消息 [session_id={session_id}, rounds={rounds}]")
        return result

    except Exception as e:
        LOG.exception(f"加载 memory 失败 [session_id={session_id}]: {e}")
        return []


def list_session_memories(session_id: str, create_dir: bool = True) -> List[Path]:
    """
    列出指定 session 的所有 memory 文件

    Args:
        session_id: 会话唯一标识符
        create_dir: 目录不存在时是否自动创建（默认 True）

    Returns:
        List[Path]: 按文件名排序的 .jsonl 文件路径列表
    """
    memory_dir = get_session_memory_dir(session_id)
    if not memory_dir.exists():
        if create_dir:
            memory_dir.mkdir(parents=True, exist_ok=True)
            LOG.info(f"已创建 memory 目录: {memory_dir}")
        return []
    return sorted(memory_dir.glob("*.jsonl"))


def get_session_history_dates(session_id: str) -> List[str]:
    """
    获取指定 session 有历史记录的日期列表

    Args:
        session_id: 会话唯一标识符

    Returns:
        List[str]: 日期字符串列表（格式：YYYY-MM-DD），按时间排序
    """
    files = list_session_memories(session_id)
    return [f.stem for f in files]
