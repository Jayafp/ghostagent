#!/usr/bin/env python3
"""
Context Optimizer 模块 - 上下文优化工具

提供三种上下文优化策略：
1. 巨大结果截断（优化一）：单个 tool_result 超过上下文窗口 30% 时进行截断
2. 历史结果压缩（优化二）：超过 N 轮的 tool_result 大于 5000 字符时进行头尾保留
3. 历史 Thinking 删除（优化三）：超过 N 轮的 assistant thinking 块删除，降低 token 消耗

用途：
- 降低长对话的上下文长度
- 避免 token 溢出
- 提高 LLM 处理效率
"""

import os
import copy
from typing import List, Dict, Set, Optional

from app.llm.utils import get_tool_use_id, is_pure_user_message
from app.log.logger import LOG

# ============ 配置常量 ============
# 优化二：N 轮之前的 tool_result 需要压缩（默认 5 轮）
TOOL_RESULT_COMPACT_ROUNDS = int(os.getenv("TOOL_RESULT_COMPACT_ROUNDS", "5"))

# 优化三：N 轮之前的 thinking 需要删除（默认 3 轮）
THINKING_REMOVE_ROUNDS = int(os.getenv("THINKING_REMOVE_ROUNDS", "3"))

# LLM 最大上下文窗口大小
LLM_MAX_WINDOW = int(os.getenv('LLM_MAX_WINDOW', '200000'))

# 优化二：触发压缩的字符数阈值
TOOL_RESULT_COMPACT_THRESHOLD = 5000

# 优化二：保留的字符数
TOOL_RESULT_COMPACT_KEEP_HEAD = 1500
TOOL_RESULT_COMPACT_KEEP_TAIL = 1500

# 优化一：单个 tool_result 占上下文的最大比例
TOOL_RESULT_MAX_RATIO = 0.30

# 跟踪已处理的消息（避免重复处理）
_processed_message_ids: Set[str] = set()


def truncate_large_tool_result(content: str, max_tokens: int) -> str:
    """
    优化一：截断过大的 tool_result 内容

    当 tool_result 内容过长时，只保留开头部分，并添加截断提示

    Args:
        content: 原始内容字符串
        max_tokens: 最大允许的 token 数

    Returns:
        str: 处理后的内容
            - 如果未超过限制，返回原始内容
            - 如果超过限制，返回截断后的内容 + 截断提示

    Note:
        token 估算使用 1 token ≈ 4 字符的粗略估算
    """
    if not content:
        return content

    # 估算当前内容的 token 数
    estimated_tokens = len(content) // 4

    if estimated_tokens <= max_tokens:
        return content

    # 需要截断，计算保留的字符数
    max_chars = max_tokens * 4
    truncated_content = content[:max_chars]

    # 添加截断提示
    truncation_notice = (
        f"\n\n...[内容已截断，原长度 {len(content)} 字符，"
        f"约 {estimated_tokens} tokens，超过上下文 {TOOL_RESULT_MAX_RATIO:.0%} 限制]"
    )

    return truncated_content + truncation_notice


def optimize_single_tool_result(block: Dict) -> Dict:
    """
    优化一：处理单个 tool_result 块

    检查 tool_result 内容是否超过单块限制，如超过则进行截断

    Args:
        block: 内容块字典，包含以下字段：
            - type: str, 必须是 "tool_result"
            - content: str, 工具结果内容

    Returns:
        Dict: 处理后的块字典
            - 如果是 tool_result 且超过限制，返回深拷贝并截断后的块
            - 否则返回原块
    """
    if not isinstance(block, dict):
        return block

    if block.get("type") != "tool_result":
        return block

    content = block.get("content", "")
    if not isinstance(content, str):
        return block

    # 计算最大允许 token 数
    max_tokens = int(LLM_MAX_WINDOW * TOOL_RESULT_MAX_RATIO)

    # 估算当前 token 数
    estimated_tokens = len(content) // 4

    if estimated_tokens > max_tokens:
        LOG.info(f"触发优化一：tool_result 内容 {estimated_tokens} tokens 超过 "
                 f"{max_tokens} tokens 限制，将进行截断")
        truncated = truncate_large_tool_result(content, max_tokens)
        block = copy.deepcopy(block)
        block["content"] = truncated

    return block


def compact_old_tool_result(content: str) -> str:
    """
    优化二：压缩历史 tool_result 内容（保留头尾）

    对于较长的历史工具结果，只保留开头和结尾，省略中间部分

    Args:
        content: 原始内容字符串

    Returns:
        str: 压缩后的内容格式：
            {头部内容}

            ...[中间部分已省略，共 {X} 字符]

            {尾部内容}

    Note:
        触发条件：内容长度 >= TOOL_RESULT_COMPACT_THRESHOLD (5000字符)
        保留长度：头部 TOOL_RESULT_COMPACT_KEEP_HEAD (1500字符)
                尾部 TOOL_RESULT_COMPACT_KEEP_TAIL (1500字符)
    """
    if not content or len(content) < TOOL_RESULT_COMPACT_THRESHOLD:
        return content

    head = content[:TOOL_RESULT_COMPACT_KEEP_HEAD]
    tail = content[-TOOL_RESULT_COMPACT_KEEP_TAIL:]

    omitted_chars = len(content) - TOOL_RESULT_COMPACT_KEEP_HEAD - TOOL_RESULT_COMPACT_KEEP_TAIL

    compacted = (
        f"{head}\n\n"
        f"...[中间部分已省略，共 {omitted_chars} 字符]\n\n"
        f"{tail}"
    )

    return compacted


def optimize_tool_results_for_llm(messages: List[Dict]) -> List[Dict]:
    """
    为 LLM 上下文优化 tool_result 内容

    策略：从后往前遍历，超过 N 轮的历史 tool_result 进行压缩

    优化点：
    1. 原地修改，避免创建新列表
    2. 已压缩的内容会记录在 _processed_message_ids 中，避免重复处理
    3. 遇到已压缩的 block_key 可以提前返回（但实现中继续遍历）

    Args:
        messages: 原始消息列表（会被原地修改）

    Returns:
        List[Dict]: 优化后的消息列表（同一列表对象）

    Processing Flow:
        1. 从后往前遍历消息
        2. 遇到纯用户消息时，轮次计数器 +1
        3. 只处理超过 TOOL_RESULT_COMPACT_ROUNDS（5轮）的 user 消息
        4. 在该消息的 content 列表中查找 tool_result 块
        5. 对超过阈值的 tool_result 块进行压缩
        6. 使用 tool_use_id 标记已处理的块
    """
    if not messages:
        return messages

    cnt_rounds = 0  # 当前是倒数第几轮

    # 从后往前遍历
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        # 检查是否是新一轮的开始（纯用户消息）
        if is_pure_user_message(msg):
            cnt_rounds += 1
            continue

        # 只处理超过 N 轮的 tool_result 消息
        if cnt_rounds < TOOL_RESULT_COMPACT_ROUNDS:
            continue

        # 只处理包含 tool_result 列表的 user 消息
        if msg.get("role") != "user":
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        # 检查该消息中的 tool_result blocks
        modified = False
        for block_index, block in enumerate(content):
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue

            block_key = get_tool_use_id(block)
            if not block_key:
                LOG.error(f"tool_result无法压缩, tool_use_id不存在: {block}")
                continue

            # 检查是否处理过
            if block_key in _processed_message_ids:
                continue

            block_content = block.get("content", "")
            if not isinstance(block_content, str):
                continue

            # 检查是否达到压缩阈值
            org_len = len(block_content)
            if org_len < TOOL_RESULT_COMPACT_THRESHOLD:
                continue

            # 执行压缩（原地修改）
            compacted = compact_old_tool_result(block_content)
            block["content"] = compacted

            # 标记为已处理
            _processed_message_ids.add(block_key)
            modified = True
            LOG.debug(f"tool_result 已经被压缩, id={block_key}, 原始长度={org_len}, "
                      f"压缩后长度={len(compacted)}")

        # 如果修改了消息但没有触发提前返回，继续往前走
        if modified:
            continue

    return messages


def optimize_tool_results_for_memory(tool_results: List[Dict]) -> List[Dict]:
    """
    为 Memory 存储优化 tool_result 内容

    应用优化一：对单个巨大的 tool_result 进行截断
    这是在将工具结果写入持久化存储前调用的

    Args:
        tool_results: tool_result 块列表，每个块是包含以下字段的字典：
            - type: str, "tool_result"
            - content: str, 结果内容
            - tool_use_id: str, 关联的工具调用ID

    Returns:
        List[Dict]: 优化后的块列表

    Note:
        与 optimize_tool_results_for_llm 不同，这里不检查 tool_use_id
        因为存储前所有内容都需要进行大小检查
    """
    if not tool_results:
        return tool_results

    optimized = []

    for i, block in enumerate(tool_results):
        if not isinstance(block, dict) or block.get("type") != "tool_result":
            optimized.append(block)
            continue

        optimized_block = optimize_single_tool_result(block)
        optimized.append(optimized_block)

    return optimized


# ============ Thinking 优化 ============

def remove_thinking_blocks(content: List[Dict]) -> List[Dict]:
    """
    从 content 列表中移除所有 thinking 块

    Args:
        content: 消息内容列表，每个元素是内容块字典

    Returns:
        List[Dict]: 移除所有 type="thinking" 块后的列表

    Example:
        >>> content = [
        ...     {"type": "thinking", "thinking": "..."},
        ...     {"type": "text", "text": "hello"}
        ... ]
        >>> remove_thinking_blocks(content)
        [{"type": "text", "text": "hello"}]
    """
    if not isinstance(content, list):
        return content

    return [block for block in content if not (isinstance(block, dict) and block.get("type") == "thinking")]


def optimize_thinking_for_llm(messages: List[Dict]) -> List[Dict]:
    """
    为 LLM 上下文优化 thinking 内容

    策略：从后往前遍历，超过 N 轮的历史 assistant 消息中的 thinking 块删除
    思考块只保留最近的几轮，更远轮次的 thinking 对当前对话价值较低，可以删除以节省 token

    优化点：
    1. 原地修改，避免创建新列表
    2. 如果 assistant 消息删除 thinking 后 content 为空，保留空列表（不影响消息结构）
    3. 使用轮次计数而非消息索引，确保语义正确

    Args:
        messages: 原始消息列表（会被原地修改）

    Returns:
        List[Dict]: 优化后的消息列表（同一列表对象）

    Processing Flow:
        1. 从后往前遍历消息
        2. 遇到纯用户消息时，轮次计数器 +1
        3. 只处理超过 THINKING_REMOVE_ROUNDS（3轮）的 assistant 消息
        4. 从该消息的 content 列表中移除所有 thinking 块
    """
    if not messages:
        return messages

    cnt_rounds = 0  # 当前是倒数第几轮

    # 从后往前遍历
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]

        # 检查是否是新一轮的开始（纯用户消息）
        if is_pure_user_message(msg):
            cnt_rounds += 1
            continue

        # 只处理超过 N 轮的 assistant 消息
        if cnt_rounds < THINKING_REMOVE_ROUNDS:
            continue

        # 只处理 assistant 角色的消息
        if msg.get("role") != "assistant":
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        # 检查是否有 thinking 块
        original_len = len(content)
        new_content = remove_thinking_blocks(content)

        if len(new_content) < original_len:
            # 原地修改 content
            msg["content"] = new_content
            removed_count = original_len - len(new_content)
            LOG.debug(f"已删除 assistant 消息中的 {removed_count} 个 thinking 块, "
                      f"索引={i}, 轮次={cnt_rounds}")

    return messages
