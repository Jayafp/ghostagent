#!/usr/bin/env python3
"""
上下文压缩模块

核心功能：
1. 当上下文接近模型限制时，保留最近 N 轮完整对话
2. 将更早的对话使用大模型压缩成摘要
3. 自适应保留轮数：确保保留的token不超过设定比率，避免频繁触发压缩

摘要包含：已完成什么、当前状态、接下来需要干什么

注意：原始对话已通过 memory_manager 持久化，压缩时无需额外归档
"""
import os
import json
import time
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

from anthropic.types import ThinkingBlock
from anthropic import Anthropic
from app.llm.utils import estimate_tokens, is_pure_user_message

from app.llm.context_optimizer import optimize_tool_results_for_llm, optimize_thinking_for_llm
from app.log.logger import LOG
from app.llm.memory_manager import MEMORY_DIR
from app.tool.task_manager import has_active_tasks, format_task_snapshot

# ============ 配置常量 ============
# 256K 上下文窗口的阈值设置
# 预留空间: 8K (max_tokens) + 4K (system) + 10K (buffer) = 22K
# 安全输入上限: 256K - 22K = 234K
#TOKEN_SOFT_LIMIT = 210000  # 软限制：约82%，较晚触发压缩，保留更多上下文
#TOKEN_HARD_LIMIT = 240000  # 硬限制：约93.7%，必须压缩，留6K余量

# 200k 上下文窗口配置:
# 200 - 22k = 178k
LLM_MAX_WINDOW = int(os.getenv('LLM_MAX_WINDOW', '200000'))
TOKEN_SOFT_LIMIT = 150000
TOKEN_HARD_LIMIT = 180000

# 保留的完整对话轮数（可从环境变量配置，默认 10）
KEEP_RECENT_ROUNDS = int(os.getenv("KEEP_RECENT_ROUNDS", "10"))

# 压缩后保留消息的token占模型最大上下文的比率（默认50%）
# 用于避免频繁触发压缩：如果KEEP_RECENT_ROUNDS轮的token超过此比率，会减少保留轮数
TOKEN_RATIO_AFTER_COMPACT = float(os.getenv("TOKEN_RATIO_AFTER_COMPACT", "0.5"))

# 摘要模型（使用轻量级模型进行压缩，节省成本）
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", os.getenv("MODEL_ID", "claude-3-haiku-20240307"))

# [Deprecated] 旧版 memories 目录，已废弃，统一使用 memory_manager.MEMORY_DIR
# 注意：KEEP_RECENT_ROUNDS 也在 memory_manager.py 中定义，避免循环导入
MEMORIES_DIR = MEMORY_DIR


def save_to_memories(messages: List[Dict], session_id: str) -> Path:
    """
    将完整对话保存到 memories 目录（归档用）

    文件名格式: memories/YYYY-MM-DD/SESSION_ID_HH-MM-SS.json

    Args:
        messages: 完整的消息列表
        session_id: 会话ID

    Returns:
        Path: 保存的文件路径
        None: 如果保存失败

    Note:
        此函数不再被主流程使用，历史消息通过 memory_manager 实时持久化
        保留用于兼容性
    """
    try:
        # 确保目录存在
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        daily_dir = MEMORIES_DIR / today
        daily_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        timestamp = datetime.datetime.now().strftime("%H-%M-%S")
        filename = f"{session_id}_{timestamp}.json"
        filepath = daily_dir / filename

        # 准备保存的数据
        save_data = {
            "session_id": session_id,
            "save_time": datetime.datetime.now().isoformat(),
            "message_count": len(messages),
            "estimated_tokens": estimate_tokens(messages),
            "messages": serialize_messages_for_save(messages)
        }

        # 写入文件
        filepath.write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        LOG.info(f"完整对话已保存到: {filepath}")
        return filepath

    except Exception as e:
        LOG.exception(f"保存 memories 失败: {e}")
        return None


def serialize_messages_for_save(messages: List[Dict]) -> List[Dict]:
    """
    将消息列表序列化为可 JSON 序列化的格式

    Args:
        messages: 消息字典列表

    Returns:
        List[Dict]: 序列化后的消息列表
    """
    result = []
    for msg in messages:
        serialized = {
            "role": msg.get("role"),
        }

        content = msg.get("content")
        if isinstance(content, list):
            serialized["content"] = [
                serialize_content_block(block) for block in content
            ]
        elif isinstance(content, str):
            serialized["content"] = content
        else:
            serialized["content"] = str(content)

        result.append(serialized)

    return result


def serialize_content_block(block: Any) -> Dict:
    """
    序列化单个内容块

    Args:
        block: 内容块对象或字典

    Returns:
        Dict: 序列化后的字典
    """
    if hasattr(block, "model_dump"):
        # Pydantic 模型
        return block.model_dump()
    elif hasattr(block, "type"):
        # Anthropic 类型
        return {
            "type": block.type,
            **extract_block_data(block)
        }
    elif isinstance(block, dict):
        return block
    else:
        return {"type": "unknown", "data": str(block)}


def extract_block_data(block: Any) -> Dict:
    """
    从 Anthropic 内容块中提取数据

    Args:
        block: Anthropic 内容块对象

    Returns:
        Dict: 包含具体数据的字典
    """
    data = {}
    if block.type == "text":
        data["text"] = getattr(block, "text", "")
    elif block.type == "thinking":
        data["thinking"] = getattr(block, "thinking", "")
    elif block.type == "tool_use":
        data["id"] = getattr(block, "id", "")
        data["name"] = getattr(block, "name", "")
        data["input"] = getattr(block, "input", {})
    elif block.type == "tool_result":
        data["tool_use_id"] = getattr(block, "tool_use_id", "")
        data["content"] = getattr(block, "content", "")
    return data


def _contains_tool_use(content: Any) -> bool:
    """
    检查消息内容是否包含 tool_use 块

    Args:
        content: 消息内容（字符串或列表）

    Returns:
        bool: 是否包含 tool_use
    """
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    return True
            elif hasattr(block, "type"):
                if block.type == "tool_use":
                    return True
    return False


def _contains_tool_result(content: Any) -> bool:
    """
    检查消息内容是否包含 tool_result 块

    Args:
        content: 消息内容（字符串或列表）

    Returns:
        bool: 是否包含 tool_result
    """
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_result":
                    return True
            elif hasattr(block, "type"):
                if block.type == "tool_result":
                    return True
    return False


def calculate_adaptive_keep_rounds(
    messages: List[Dict],
    max_rounds: int = KEEP_RECENT_ROUNDS,
    ratio: float = TOKEN_RATIO_AFTER_COMPACT,
    max_window: int = LLM_MAX_WINDOW
) -> int:
    """
    自适应计算需要保留的轮数

    策略：
    1. 先尝试保留 max_rounds 轮，估算其 token 数
    2. 如果超过比率限制（ratio * max_window），则逐步减少轮数
    3. 最终保留的轮数 = min(max_rounds, 比率限制下的最大轮数)

    Args:
        messages: 完整消息列表
        max_rounds: 期望保留的最大轮数（来自KEEP_RECENT_ROUNDS）
        ratio: 压缩后保留token占最大窗口的比率
        max_window: 模型最大上下文窗口大小

    Returns:
        实际应该保留的轮数
    """
    # 空消息列表直接返回1（最少保留1轮）
    if not messages:
        return 1

    target_tokens = int(max_window * ratio)

    # 从少到多尝试，找到满足比率限制的最大轮数
    # 使用二分查找优化
    low, high = 1, max_rounds
    best_rounds = 1

    while low <= high:
        mid = (low + high) // 2
        keep_msgs, _ = split_messages_by_rounds(messages, mid)
        keep_tokens = estimate_tokens(keep_msgs)

        if keep_tokens <= target_tokens:
            # 满足限制，尝试更多轮数
            best_rounds = mid
            low = mid + 1
        else:
            # 超出限制，减少轮数
            high = mid - 1

    if best_rounds < max_rounds:
        LOG.info(f"自适应调整保留轮数: {max_rounds} -> {best_rounds} (目标token: {target_tokens}, 比率: {ratio:.0%})")

    return best_rounds


def split_messages_by_rounds(messages: List[Dict], keep_rounds: int) -> Tuple[List[Dict], List[Dict]]:
    """
    将消息按对话轮数分割，确保 tool_use/tool_result 对不被拆散

    一轮对话 = user 消息 + assistant 消息（可能包含 tool_use）+ tool_result 消息

    重要：如果一个 assistant 消息包含 tool_use，那么紧接着的 user 消息（包含 tool_result）
    必须与之保持在同一侧，不能分割开，否则会导致 API 调用失败。

    Args:
        messages: 完整消息列表
        keep_rounds: 需要保留的最近轮数

    Returns:
        (保留的消息列表, 需要压缩的消息列表)
    """
    if not messages:
        return [], []

    # 从后向前遍历，标记每个消息属于第几轮
    # 策略：包含 tool_result 的 user 消息属于 assistant's turn 的延续，
    # 不应增加轮次计数
    # 结果：message_rounds[i] 越小，表示消息越新（从新往旧数第几轮）
    message_rounds = [0] * len(messages)
    round_count = 0

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # 每个不包含 tool_result 的 user 消息标志新一轮的开始
        if msg.get("role") == "user" and not _contains_tool_result(msg.get("content")):
            round_count += 1
        message_rounds[i] = round_count

    # 找到分割点：保留最近 keep_rounds 轮
    # 策略：从后往前找到第 keep_rounds 个普通 user 消息（轮次起点）
    split_index = 0  # 默认保留所有
    rounds_found = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        # 只有普通 user 消息（非 tool_result）才标志一个轮次的起点
        if msg.get("role") == "user" and not _contains_tool_result(msg.get("content")):
            rounds_found += 1
            if rounds_found == keep_rounds:
                split_index = i
                break

    # 关键修复：处理边界情况
    # 如果分割点正好在一个包含 tool_use 的 assistant 消息之前，
    # 我们必须把对应的包含 tool_result 的 user 消息也保留下来
    if split_index > 0 and split_index < len(messages):
        prev_msg = messages[split_index - 1]
        if prev_msg.get("role") == "assistant" and _contains_tool_use(prev_msg.get("content")):
            # 向前找包含 tool_result 的 user 消息
            for j in range(split_index, len(messages)):
                if messages[j].get("role") == "user" and _contains_tool_result(messages[j].get("content")):
                    # 调整分割点，将 tool_use 和 tool_result 都保留
                    split_index = j + 1
                    break

    keep_messages = messages[split_index:]
    compress_messages = messages[:split_index]

    return keep_messages, compress_messages


def format_message_for_summary(msg):
    # role = msg.get("role", "unknown")
    # content = msg.get("content")
    #
    # lines = []
    # if role == "user":
    #     lines.append(f"{format_content(content)}")
    # elif role == "assistant":
    #     lines.append(f"{format_content(content)}")
    # else:
    #     lines.append(f"{format_content(content)}")
    #
    # return "\n".join(lines)
    return format_content(msg.get("role"), msg.get("content"))

def format_messages_for_summary(messages: List[Dict]) -> str:
    """
    将消息格式化为便于模型理解的文本格式
    """
    lines = []

    for msg in messages:
        lines.append(format_message_for_summary(msg))

    return "\n".join(lines)


def format_content(role: str, content: Any) -> str:
    """
    格式化内容
    """
    if isinstance(content, str):
        if role == 'user':
            return f'[用户提问]: {content}'
        else:
            LOG.warning(f"消息格式有问题??? content 类型是string, role={role}, ")
            return f'{content}'
        #return content[:1000] + "..." if len(content) > 1000 else content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "unknown")
                if block_type == "text":
                    parts.append(f"[助理回复]: {block.get("text", "")}")
                elif block_type == "thinking":
                    parts.append(f"[助理思考]: {block.get('thinking', '')[:200]}...")
                elif block_type == "tool_use":
                    parts.append(f"[调用工具]: {block.get('name', '')}({block.get('input', {})})")
                elif block_type == "tool_result":
                    result = block.get("content", "")
                    parts.append(f"[工具结果]: {result[:200]}...")
            elif hasattr(block, "type"):
                # Anthropic 对象
                if block.type == "text":
                    parts.append(f"[助理回复]: {getattr(block, "text", "")}")
                elif block.type == "thinking":
                    parts.append(f"[助理思考]: {getattr(block, 'thinking', '')[:200]}...")
                elif block.type == "tool_use":
                    parts.append(f"[调用工具]: {getattr(block, 'name', '')}")
                elif block.type == "tool_result":
                    parts.append(f"[工具结果]: {getattr(block, 'content', '')[:200]}")
        return "\n".join(parts)

    #return str(content)[:500]
    return str(content)


def generate_summary(messages: List[Dict], api_key: str = None, base_url: str = None) -> str:
    """
    使用大模型生成对话摘要

    摘要必须包含：
    1. 已完成什么
    2. 当前状态
    3. 接下来需要干什么

    Args:
        messages: 需要压缩的消息列表
        api_key: API key (可选，默认从环境变量读取)
        base_url: API base url (可选)

    Returns:
        生成的摘要文本
    """
    # todo 如果需要压缩的 meesages 太大, 导致token溢出, 可能会压缩失败, 要处理
    try:
        # 初始化客户端
        if api_key is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        if base_url is None:
            base_url = os.getenv("ANTHROPIC_BASE_URL")

        client = Anthropic(api_key=api_key, base_url=base_url)

        # 准备提示
        conversation_text = format_messages_for_summary(messages)

        summary_prompt = f"""你是一个对话总结专家。请将以下对话内容总结成一段简洁的摘要。

**要求：**
1. 总结必须包含以下三个方面：
   - 聚焦事实性内容：讨论了什么、做出了哪些决策、当前状态如何。
   - 保持必需的摘要结构和章节标题不变。
   - 不要翻译或修改代码、文件路径、标识符或错误信息。

2. 保持客观，不要添加对话中没有的信息。

3. 保留所有不透明标识符的原始写法（不得缩短或重构），包括 UUID、哈希值、ID、令牌、API 密钥、主机名、IP 地址、端口、URL 和文件名。

4. 如果对话中使用了任务管理工具（task_create / list_task / update_task / complete_task / finish_task），必须在摘要中保留任务图的关键状态：哪些任务已完成、哪个进行中、哪些待办及其依赖关系、以及关键进度笔记。这些信息对后续继续任务至关重要，不得省略。

**对话内容：**
{conversation_text}

**请输出简洁的摘要（不超过800字）：**"""

        # 调用模型生成摘要
        response = client.messages.create(
            model=SUMMARY_MODEL,
            max_tokens=2048,
            system="你是一个专业的对话总结助手，擅长提取关键信息并生成结构化的摘要。",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.3  # 降低随机性，保持摘要的稳定性
        )

        summary = "[摘要生成失败]"
        for block in response.content:
            if block and hasattr(block, 'text'):
                summary = block.text
                break
        # 如果有开启thinking, content[0]可能没有text
        #summary = response.content[0].text if response.content else "[摘要生成失败]"
        LOG.info(f"摘要生成成功，长度: {len(summary)} 字符")
        return summary

    except Exception as e:
        LOG.exception(f"生成摘要失败: {e}")
        # 如果摘要生成失败，返回一个简单的占位符
        return f"[之前的对话内容已归档，共 {len(messages)} 条消息]"


def create_summary_message(summary: str, session_id: str = None) -> Dict:
    """
    创建摘要消息对象。

    若 session 存在未结束的任务计划，在摘要末尾追加当前任务进度快照
    （来自 task_manager 的权威状态，不依赖摘要模型是否真的保留了任务信息），
    以保证上下文压缩后 agent 仍能继续未完成的任务。
    """
    content = f"""📋 **[历史对话摘要]**

{summary}
"""
    if session_id:
        try:
            if has_active_tasks(session_id):
                snapshot = format_task_snapshot(session_id, include_notes=True)
                if snapshot:
                    content += f"\n---\n## 📌 当前任务进度（任务图 DAG，权威状态）\n\n{snapshot}\n"
        except Exception as e:
            LOG.warning(f"附加任务进度快照失败 [{session_id}]: {e}")

    content += f"\n---\n*(以上为之前对话的压缩摘要，详细内容已保存到 memory 目录)*"
    return {"role": "assistant", "content": content}


def smart_compact(
    messages: List[Dict],
    session_id: str = None,
    force: bool = False
) -> List[Dict]:
    """
    智能压缩上下文

    策略：
    1. 估算当前 token 数
    2. 如果超过软限制，或强制压缩，则执行压缩
    3. 保留最近 N 轮完整对话
    4. 将更早的对话用大模型生成摘要
    5. 返回：[摘要消息] + [保留的完整对话]

    极端情况处理：
    - 如果保留的 N 轮仍超过硬限制，逐步减少保留轮数直到满足限制
    - 如果减少到只剩 1 轮仍超限，则对保留轮次进行截断处理

    注意：原始对话已通过 memory_manager 实时持久化，压缩时无需额外归档

    Args:
        messages: 当前消息列表
        session_id: 会话ID（保留参数用于兼容，但不再用于归档）
        force: 是否强制压缩

    Returns:
        压缩后的消息列表
    """
    if not messages:
        return messages

    # 压缩之前, 先处理 tool_result
    messages = optimize_tool_results_for_llm(messages)

    # 删除超过 N 轮的历史 thinking 块
    messages = optimize_thinking_for_llm(messages)

    current_tokens = estimate_tokens(messages)
    LOG.debug(f"当前上下文 token 数: {current_tokens}")

    # 检查是否需要压缩
    if not force and current_tokens < TOKEN_SOFT_LIMIT:
        LOG.debug("未达到压缩阈值，跳过")
        return messages

    LOG.info(f"触发上下文压缩，当前 {current_tokens} tokens")
    start = time.perf_counter()

    # 步骤1: 自适应计算需要保留的轮数
    # 避免保留过多导致下次又触发压缩
    adaptive_rounds = calculate_adaptive_keep_rounds(
        messages,
        max_rounds=KEEP_RECENT_ROUNDS,
        ratio=TOKEN_RATIO_AFTER_COMPACT,
        max_window=LLM_MAX_WINDOW
    )

    # 步骤2: 分割消息 - 保留计算出的轮数，压缩更早的
    keep_messages, compress_messages = split_messages_by_rounds(messages, adaptive_rounds)

    if not compress_messages:
        LOG.info("没有需要压缩的历史消息")
        return messages

    LOG.info(f"保留最近 {adaptive_rounds} 轮对话 ({len(keep_messages)} 条)，压缩更早的 {len(compress_messages)} 条")

    # 步骤3: 生成摘要
    summary = generate_summary(compress_messages)
    summary_message = create_summary_message(summary, session_id=session_id)

    # 步骤4: 组装新的消息列表
    new_messages = [summary_message] + keep_messages
    new_tokens = estimate_tokens(new_messages)

    # 步骤5: 处理极端情况 - 即使按比例保留仍超过硬限制（后备保护）
    if new_tokens > TOKEN_HARD_LIMIT:
        LOG.warning(f"保留 {adaptive_rounds} 轮后仍超过硬限制 ({new_tokens} > {TOKEN_HARD_LIMIT})，进入紧急压缩模式")
        new_messages = _emergency_compact(summary_message, keep_messages, compress_messages, session_id=session_id)

    final_tokens = estimate_tokens(new_messages)
    LOG.info(f"压缩完成: {current_tokens} -> {final_tokens} tokens，减少 {current_tokens - final_tokens} tokens, cost: {(time.perf_counter() - start):.3f}s")

    return new_messages


def _emergency_compact(
    summary_message: Dict,
    keep_messages: List[Dict],
    compress_messages: List[Dict],
    session_id: str = None
) -> List[Dict]:
    """
    紧急压缩模式：当标准压缩后仍超过硬限制时调用

    策略（渐进式）：
    1. 逐步减少保留轮数（从 KEEP_RECENT_ROUNDS 递减到 1）
    2. 如果减少到 1 轮仍超限，则对保留的消息进行截断
    3. 最终保底：只返回摘要 + 最近一轮的最后几条消息

    Args:
        summary_message: 已生成的摘要消息
        keep_messages: 原本要保留的消息
        compress_messages: 已被压缩成摘要的消息（用于合并摘要）

    Returns:
        进一步压缩后的消息列表
    """
    # 策略1: 逐步减少保留轮数
    for rounds in range(KEEP_RECENT_ROUNDS - 2, 0, -2):
        reduced_keep, additional_compress = split_messages_by_rounds(keep_messages, rounds)

        if additional_compress:
            # 将额外需要压缩的消息合并到原压缩消息中，重新生成摘要
            all_compress = compress_messages + additional_compress
            combined_summary = generate_summary(all_compress)
            combined_summary_message = create_summary_message(combined_summary, session_id=session_id)
        else:
            combined_summary_message = summary_message
            reduced_keep = keep_messages

        candidate_messages = [combined_summary_message] + reduced_keep
        candidate_tokens = estimate_tokens(candidate_messages)

        LOG.info(f"尝试保留 {rounds} 轮: {candidate_tokens} tokens")

        if candidate_tokens <= TOKEN_HARD_LIMIT:
            LOG.info(f"紧急压缩成功：保留最近 {rounds} 轮对话")
            return candidate_messages

    # 策略2: 极端情况 - 即使只保留 1 轮也超限
    # 将 keep_messages（1轮）也合并到摘要中，只保留最后一轮的最后部分
    LOG.warning(f"即使保留 1 轮仍超过硬限制，进行最终截断处理")

    # 取最后一轮的最后一条用户消息和助手回复（如果有）
    last_messages = _extract_last_user_exchange(keep_messages)

    # 重新生成包含保留消息的摘要
    all_messages_for_summary = compress_messages + keep_messages
    final_summary = generate_summary(all_messages_for_summary)
    final_summary_message = create_summary_message(final_summary, session_id=session_id)

    if last_messages:
        final_messages = [final_summary_message] + last_messages
        final_tokens = estimate_tokens(final_messages)
        LOG.warning(f"最终截断：仅保留摘要 + {len(last_messages)} 条最新消息，共 {final_tokens} tokens")
        return final_messages

    # 保底：只返回摘要
    LOG.error(f"极端情况：上下文过大，仅保留摘要消息")
    return [final_summary_message]


def _extract_last_user_exchange(messages: List[Dict]) -> List[Dict]:
    """
    提取最后一轮中最后的关键消息（用户消息 + 可能的助手回复）

    重要：确保不会破坏 tool_use/tool_result 对。如果包含 tool_use 的助手消息被保留，
    那么对应的包含 tool_result 的用户消息也必须被保留。

    用于极端情况下保留最基本的上下文连续性
    """
    if not messages:
        return []

    # 从后往前找，找到最后一条用户消息（不包括只包含 tool_result 的消息）
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if is_pure_user_message(msg):
            last_user_idx = i
            break

        # if msg.get("role") == "user":
        #     content = msg.get("content")
        #     # 如果这个 user 消息只包含 tool_result，它不是新一轮的开始
        #     # 我们需要继续往前找真正的用户输入
        #     if _contains_tool_result(content) and not _is_pure_text_user_message(content):
        #         continue
        #     last_user_idx = i
        #     break

    if last_user_idx == -1:
        # 没有找到真正的用户消息，可能全是 tool_result
        # 这种情况下保留所有消息（不应该压缩）
        return messages

    # 包含最后一条用户消息及其后的所有消息
    result = messages[last_user_idx:]

    # 安全检查：确保没有不完整的 tool_use/tool_result 对
    # 如果结果中的第一条消息只包含 tool_result，但前一条消息（不在结果中）不包含 tool_use
    # 这会导致不完整的对。我们需要扩大范围包含前面的 assistant 消息
    if result and result[0].get("role") == "user":
        first_content = result[0].get("content")
        if _contains_tool_result(first_content) and last_user_idx > 0:
            prev_msg = messages[last_user_idx - 1]
            if prev_msg.get("role") == "assistant" and _contains_tool_use(prev_msg.get("content")):
                # 需要把前面的 assistant 消息也包含进来
                result = [prev_msg] + result

    # 如果消息太长，截断内容
    truncated_result = []
    for msg in result:
        truncated_msg = _truncate_message_content(msg, max_length=2000)
        truncated_result.append(truncated_msg)

    return truncated_result


# def _is_pure_text_user_message(content: Any) -> bool:
#     """
#     检查用户消息是否只包含纯文本（不包含 tool_result）
#     """
#     if isinstance(content, str):
#         return True
#     if isinstance(content, list):
#         for block in content:
#             if isinstance(block, dict):
#                 if block.get("type") == "tool_result":
#                     return False
#             elif hasattr(block, "type"):
#                 if block.type == "tool_result":
#                     return False
#         # list 中全是 text/thinking 等非 tool_result
#         return True
#     return True


def _truncate_message_content(message: Dict, max_length: int = 2000) -> Dict:
    """
    截断消息内容到指定长度
    """
    import copy
    msg = copy.deepcopy(message)
    content = msg.get("content")

    if isinstance(content, str) and len(content) > max_length:
        msg["content"] = content[:max_length] + f"\n...[内容已截断，原长度 {len(content)} 字符]"
    elif isinstance(content, list):
        # 截断列表中的文本块
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if len(text) > max_length:
                    block["text"] = text[:max_length] + f"\n...[内容已截断，原长度 {len(text)} 字符]"

    return msg


# ============ 向后兼容的旧函数 ============

def _compact_tool_result(messages: list) -> list:
    """
    旧版函数：保留最近几个 tool_result
    现在由 smart_compact 替代
    """
    return messages


def _compact_thinking(messages: list):
    """
    旧版函数：保留最近几个 thinking
    现在由 smart_compact 替代
    """
    return messages


def micro_compact(messages: list, session_id: str = None) -> list:
    """
    保持原有接口，内部调用新的智能压缩

    Args:
        messages: 消息列表
        session_id: 会话ID（可选，用于保存 memories）

    Returns:
        压缩后的消息列表
    """
    return smart_compact(messages, session_id=session_id, force=False)
